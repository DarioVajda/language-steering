"""Check of the tuple-of-layers path of Model.generate (../mgsm_steering/generate.py) used by the every-layer arms
of recover.py: with one layer given as a tuple, (l,), the generated tokens must be identical to the single-layer
path (layer=l) that produced our instruction + steering runs, all-token and prompt-only. Also prints one every-layer
response per alpha as a sanity check. A few MGSM dev prompts, 128 new tokens.

Usage: XL_MODEL=<model> python every_layer_check.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
import generate as G  # noqa: E402
from generate_pairs import GEN, MODEL, MODELS, Model, prompt, split  # noqa: E402

G.MAX_NEW = 128
M = Model(MODELS[MODEL]["hf"])
mu = torch.load(GEN / "recover" / "vectors.pt")
pairs = [("de", "ja"), ("zh", "ru"), ("sw", "fr"), ("ja", "de")]
items = [(r, s, t) for s, t in pairs for r in split()["dev"][:4]]
texts = [prompt("instruction", r, s, t) for r, s, t in items]
l = MODELS[MODEL]["layers"][2]
ok = True
for po in (False, True):
    v = [1.0 * (mu[t][l] - mu[s][l]) for _, s, t in items]
    a = M.generate(texts, l, v, po, return_ids=True)
    b = M.generate(texts, (l,), [x[None] for x in v], po, return_ids=True)
    same = sum(x[-1] == y[-1] for x, y in zip(a, b))
    ok &= same == len(a)
    print(f"layer {l}, prompt_only={po}: identical token ids {same}/{len(a)}", flush=True)
n = len(M.blocks)
for alpha in (0.02, 0.1, 0.5):
    out = M.generate(texts[:2], tuple(range(1, n)), [alpha * (mu[t] - mu[s])[1:n] for _, s, t in items[:2]], True)
    print(f"every layer, prompt-only, alpha {alpha}: {out[0][0][:100]!r}", flush=True)
print("CHECK", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
