"""J1: score a clip with Cosmos 3 as a judge. Score = logprob("yes") - logprob("no").

Model choice. Cosmos 3 is an OMNIMODAL world model: the same nvidia/Cosmos3-Nano weights we
generate video with also serve as the reasoner, exposed in the model card as
`Cosmos3ReasonerForConditionalGeneration` with "Reasoner Input: Text+Video, mp4 at the
recommended 4 fps". So the judge needs no extra download and shares a physics prior with
the generator. (The older Cosmos-Reason1-7B / Reason2-8B line is a separate, smaller
family; not used here.)

Serving. The card documents a vLLM server, but the vllm in this env is a CUDA-13 build
against a cu128 torch and does not import. `transformers` exposes the omni model directly
as `Cosmos3OmniForConditionalGeneration`, which is better for our purpose anyway: we need
the logits at one position, not sampled text, and a forward pass gives them without a
server or a decode loop.

Scoring, per the project spec rather than the card: a fixed yes/no question in
DIRECT-ANSWER form (no chain-of-thought), read as the difference of the two logprobs at the
first answer token. Chain-of-thought would make the score depend on sampled reasoning text,
which is neither deterministic nor a scalar. Averaged over three fixed paraphrases to blunt
prompt sensitivity.
"""
import numpy as np

MODEL = "nvidia/Cosmos3-Nano"

# Three fixed paraphrases. Same question, different surface form; the spread across them is
# reported so prompt sensitivity is visible rather than hidden.
PARAPHRASES = (
    "Does the cloth in this video move like {c}? Answer yes or no.",
    "Watch the fabric. Is this motion consistent with {c}? Answer yes or no.",
    "Judging only by how it moves and folds, is this {c}? Answer yes or no.",
)

SYSTEM = ("You are a physics-aware video analyst. Answer the question with a single word, "
          "either yes or no. Do not explain.")


class CosmosJudge:
    def __init__(self, model_id=MODEL, device="cuda:0", fps=4.0, max_pixels=None):
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
        # The answer token may be "yes" or " yes" depending on what precedes it; take
        # whichever variant the tokenizer actually produces as a single leading token.
        self.yes_ids = self._variants(tok, "yes")
        self.no_ids = self._variants(tok, "no")
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError("could not resolve yes/no token ids")

    @staticmethod
    def _variants(tok, word):
        out = []
        for s in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                out.append(ids[0])
        return sorted(set(out))

    def _messages(self, clip_path, question):
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": f"file://{clip_path}", "fps": self.fps},
                {"type": "text", "text": question},
            ]},
        ]

    def _decode(self, clip_path):
        """Decode the mp4 ourselves, subsampled to the card's recommended 4 fps.

        qwen_vl_utils cannot read video in this environment: its torchcodec backend fails
        to load (CUDA-13 libs against a cu128 torch) and the torchvision fallback calls
        `torchvision.io.read_video`, which no longer exists in this version. Left to a bare
        except that returned no video, the judge scored the TEXT ALONE -- giving byte
        identical scores for three visibly different clips. Decoding here removes the
        dependency and makes the failure impossible to hide.
        """
        import imageio.v2 as imageio
        import numpy as np

        rd = imageio.get_reader(str(clip_path))
        src_fps = float(rd.get_meta_data().get("fps", 30.0))
        step = max(int(round(src_fps / self.fps)), 1)
        frames = [f[..., :3] for i, f in enumerate(rd) if i % step == 0]
        rd.close()
        if not frames:
            raise RuntimeError(f"no frames decoded from {clip_path}")
        return np.stack(frames)

    def score_one(self, clip_path, question):
        """logprob(yes) - logprob(no) at the first answer token."""
        torch = self.torch
        msgs = self._messages(clip_path, question)
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        video = self._decode(clip_path)
        inputs = self.processor(text=[text], images=None, videos=[video],
                                return_tensors="pt").to(self.device)
        # The clip must actually be in the prompt. Without this the judge silently
        # degenerates into a text-only prior that is identical for every rollout.
        if not hasattr(self, "_checked"):
            n_tok = int(inputs["input_ids"].shape[-1])
            has_pix = any(k.startswith("pixel_values") for k in inputs)
            if not has_pix or n_tok < 64:
                raise RuntimeError(
                    f"video did not reach the model (tokens={n_tok}, keys={list(inputs)})")
            self._checked = {"tokens": n_tok, "frames": int(video.shape[0])}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        y = torch.logsumexp(logp[self.yes_ids], dim=0).item()
        n = torch.logsumexp(logp[self.no_ids], dim=0).item()
        return y - n, y, n

    def score(self, clip_path, material):
        """Average the yes-no margin over the fixed paraphrases."""
        vals = []
        for p in PARAPHRASES:
            s, _y, _n = self.score_one(clip_path, p.format(c=material))
            vals.append(s)
        a = np.asarray(vals, float)
        return {"score": float(a.mean()), "spread": float(a.max() - a.min()),
                "per_prompt": [float(x) for x in a]}
