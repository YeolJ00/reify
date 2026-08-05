"""Pairwise comparator: "which of these two moves more like {material}?"

Rule 1 exists because absolute yes/no scoring failed. Two clips go into ONE prompt and the
score is the A-vs-B logprob margin, so any per-clip offset -- the judge's baseline opinion
of the object, the render, the framing -- cancels inside the comparison rather than being
averaged over afterwards.

Position bias is real and is removed by construction: every pair is asked in both orders
and the margin is s = (s_XY - s_YX) / 2. The leftover (s_XY + s_YX) / 2 is the bias itself
and is reported, not discarded, because a large bias means the comparison is being decided
by slot rather than by content.

No-think scoring (G0). Cosmos3-Nano's chat template contains no thinking/reasoning block
and its config exposes no reasoning flag, so there is nothing to switch off -- and we never
sample anyway. The score is read from the logits at the first answer position, which is a
deterministic function of the inputs. A sampled trace would have neither property, which is
also why the G-track needs this mode: you cannot differentiate through a sampled trace.
"""
import numpy as np

MODEL = "nvidia/Cosmos3-Nano"

# Two surface forms of the same comparison. Kept few on purpose: the pairwise margin is
# meant to remove the prompt sensitivity that three paraphrases failed to average out in
# the absolute-score version.
PROMPTS = (
    "Which clip moves more like {material}? Answer A or B.",
    "One of these is {material}. Judging only by the motion, which one? Answer A or B.",
)

SYSTEM = ("You compare two videos of the same object. Answer with a single letter, "
          "A or B. Do not explain.")


class PairwiseJudge:
    def __init__(self, model_id=MODEL, device="cuda:0", fps=4.0):
        import torch
        from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

        self.torch = torch
        self.fps = float(fps)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Cosmos3OmniForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device)
        self.model.eval()
        self.device = device
        tok = self.processor.tokenizer
        self.a_ids = self._variants(tok, "A")
        self.b_ids = self._variants(tok, "B")
        if not self.a_ids or not self.b_ids:
            raise RuntimeError("could not resolve A/B token ids")
        self._cache = {}
        self._checked = None

    @staticmethod
    def _variants(tok, w):
        out = []
        for s in (w, " " + w, w.lower(), " " + w.lower()):
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                out.append(ids[0])
        return sorted(set(out))

    def _decode(self, path):
        """Own decoder, subsampled to the card's 4 fps. Cached: the kill test reuses each
        clip across many pairs, and decoding dominates otherwise."""
        key = str(path)
        if key in self._cache:
            return self._cache[key]
        import imageio.v2 as imageio
        rd = imageio.get_reader(key)
        src = float(rd.get_meta_data().get("fps", 30.0))
        step = max(int(round(src / self.fps)), 1)
        frames = [f[..., :3] for i, f in enumerate(rd) if i % step == 0]
        rd.close()
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        v = np.stack(frames)
        self._cache[key] = v
        return v

    def _one(self, vA, vB, question):
        """logprob(A) - logprob(B) at the first answer token, for this exact ordering."""
        torch = self.torch
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Clip A:"},
                {"type": "video"},
                {"type": "text", "text": "Clip B:"},
                {"type": "video"},
                {"type": "text", "text": question},
            ]},
        ]
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=None, videos=[vA, vB],
                                return_tensors="pt").to(self.device)
        if self._checked is None:
            n_tok = int(inputs["input_ids"].shape[-1])
            grid = inputs.get("video_grid_thw")
            n_vid = int(grid.shape[0]) if grid is not None else 0
            # Both clips must actually be in the prompt. With one video the comparison
            # silently becomes a one-clip prior and every pair returns the same margin.
            if n_vid != 2 or n_tok < 256:
                raise RuntimeError(
                    f"expected 2 videos in the prompt, got {n_vid} (tokens={n_tok})")
            self._checked = {"tokens": n_tok, "videos": n_vid,
                             "frames": int(vA.shape[0])}
        with torch.no_grad():
            # Only the last position is ever read, so run the LM head there alone. With
            # two videos the prompt is ~1044 tokens and a ~150k vocab, so the full logits
            # tensor is a ~600 MB allocation per call -- enough to OOM on a shared GPU,
            # and pure waste. This also keeps the graph small for the G1 gradient pass.
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        a = torch.logsumexp(lp[self.a_ids], dim=0).item()
        b = torch.logsumexp(lp[self.b_ids], dim=0).item()
        return a - b

    def compare(self, path_x, path_y, material, prompts=PROMPTS):
        """Order-averaged preference for X over Y. Positive means X wins.

        Returns margin, the position bias that was removed, and the per-prompt margins so
        prompt disagreement stays visible rather than hidden inside a mean.
        """
        vx, vy = self._decode(path_x), self._decode(path_y)
        margins, biases = [], []
        for p in prompts:
            q = p.format(material=material)
            s_xy = self._one(vx, vy, q)      # X shown as A
            s_yx = self._one(vy, vx, q)      # Y shown as A
            margins.append(0.5 * (s_xy - s_yx))
            biases.append(0.5 * (s_xy + s_yx))
        m = np.asarray(margins, float)
        return {"margin": float(m.mean()),
                "spread": float(m.max() - m.min()) if len(m) > 1 else 0.0,
                "bias": float(np.mean(biases)),
                "per_prompt": [float(v) for v in m]}
