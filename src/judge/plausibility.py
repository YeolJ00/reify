"""Absolute physical-plausibility judge, Cosmos 3 cookbook form.

Difference from everything tried before, and why each part matters:

* It NAMES the principles to check -- object permanence, shape constancy, continuous
  trajectories -- instead of asking a bare "is this realistic". Every earlier prompt left
  the model to decide what "realistic" meant, and it settled on "moves a lot"
  (rho(preference, motion magnitude) = +0.85).
* It explicitly says to ignore simulation/render quality. Our clips are Cycles renders of
  a Blender scene; without this the model can answer "looks CG" instead of judging events,
  which is a confound no amount of render work removes.
* The answer is a forced (A)/(B) choice with the options written out, so the score is the
  logit margin at one position -- deterministic, differentiable, no sampled trace.

Score = logprob(A) - logprob(B) = logprob(possible) - logprob(impossible). Positive means
the model judges the clip physically possible.
"""
import numpy as np

MODEL = "nvidia/Cosmos3-Nano"

QUESTION = (
    "Is this video physically plausible/possible according to your understanding of e.g. "
    "object permanence, shape constancy (objects maintain shape over time), continuous "
    "trajectories of objects? Assume it is the normal laws of physics.\n"
    "Your answer should be based on the events in the video and ignore the quality of the "
    "simulation engine.\n"
    "(A) Possible\n"
    "(B) Impossible"
)

SYSTEM = "Answer with a single letter, A or B. Do not explain."


class PlausibilityJudge:
    def __init__(self, model_id=MODEL, device="cuda:0", n_frames=None):
        import torch
        from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

        self.torch = torch
        self.n_frames = n_frames          # None = every frame in the clip
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Cosmos3OmniForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device)
        self.model.eval()
        self.device = device
        tok = self.processor.tokenizer
        self.a_ids = self._variants(tok, "A")
        self.b_ids = self._variants(tok, "B")
        self._checked = None

    @staticmethod
    def _variants(tok, w):
        out = []
        for s in (w, " " + w, w.lower(), " " + w.lower()):
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                out.append(ids[0])
        return sorted(set(out))

    def _frames(self, path):
        import imageio.v2 as imageio
        rd = imageio.get_reader(str(path))
        f = [x[..., :3] for x in rd]
        rd.close()
        v = np.stack(f)
        if self.n_frames and len(v) > self.n_frames:
            idx = np.unique(np.linspace(0, len(v) - 1, self.n_frames).round().astype(int))
            v = v[idx]
        return v

    def score(self, path, question=QUESTION):
        """logprob(possible) - logprob(impossible). Positive = judged possible."""
        torch = self.torch
        v = self._frames(path)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "video"},
                                             {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        # do_sample_frames=False or the processor silently clamps to min_frames=4
        inputs = self.processor(text=[text], images=None, videos=[v],
                                return_tensors="pt",
                                do_sample_frames=False).to(self.device)
        if self._checked is None:
            g = inputs.get("video_grid_thw")
            self._checked = {"tokens": int(inputs["input_ids"].shape[-1]),
                             "frames_in": int(v.shape[0]),
                             "frames_seen": int(g[0][0]) * 2 if g is not None else 0}
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, logits_to_keep=1)
            h = out.hidden_states[-1][0, -1].float()
            logits = h @ self.model.lm_head.weight.float().T   # fp32 head, see pairwise.py
        lp = torch.log_softmax(logits, dim=-1)
        a = torch.logsumexp(lp[self.a_ids], dim=0).item()
        b = torch.logsumexp(lp[self.b_ids], dim=0).item()
        return a - b
