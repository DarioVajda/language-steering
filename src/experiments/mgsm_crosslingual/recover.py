"""ReCoVeR (Sterz, Schmidt, Glavaš, Vulić, Findings of EMNLP 2025; code github.com/hSterz/recover @ 4c6eb4d) as a
baseline on cross-lingual MGSM: its training-free ("naive", unsupervised) steering, reimplemented in
Model.generate (../mgsm_steering/generate.py, recover=...; checked against their code by recover_check.py), on top
of the instruction prompt, against our instruction + prompt-only steering at one layer. ReCoVeR+ (trained) is not run.

ReCoVeR adds alpha * (r_T - r_S)(j) / |(r_T - r_S)(j)| to the output of every block j, at every position but the first
token of the prompt (the chat template included) and at every generated token, with the norm restore of all their
published runs (--restore_norm: each output rescaled to the norm of the unsteered one). r_L(j) = mu_L(j) - (the mean over languages), mu_L(j) the
mean hidden state j over the tokens (the first excluded) of FLORES sentences in L; the centring cancels in r_T - r_S,
so the vector is mu_T(j) - mu_S(j), from the same 997 FLORES-200 dev sentences and Model.mean_states as ours. As in
their code, the vector of the last block is the mean of the model's last hidden state, which Hugging Face returns
after the final norm.

The one arm, recover, is ReCoVeR as published, on the instruction prompt (the S question + "Answer in <T>." in S,
the `instruction` arm). Its only tuned setting is alpha (their LCB runs use 1, MultiQ 2), in units of the normalised
vector: BASE_ALPHAS, extended by EXTEND doublings of the largest alpha while the best one is the largest (the scale
of the hidden states differs between models). It is chosen on the dev items of the 11 tuning pairs by the rule of
analyse_pairs.py (highest mean success; ties to the smaller alpha), then run on the test items of the 99 held-out
pairs. Prompts, decoding (greedy, at most 2048 new tokens), batching, split and scoring are those of
generate_pairs.py.

Tasks per model, shared by any number of workers through claim files (one per task, under
results/<model>/generations/claims/recover/), in order:
  vectors   mu_L(j) for every hidden state j and the 11 languages -> generations/recover/vectors.pt
  dev       one unit per alpha over the 11 tuning pairs -> generations/recover/dev/recover_a<alpha>_r1.jsonl
  select    -> generations/recover/chosen.json
  test      one unit per held-out pair -> generations/recover/test/recover_<S>-<T>.jsonl
  analyse   against the existing test runs of the instruction, our arms and native -> results/<model>/recover.json
A worker heartbeats its claim (touches it every minute); a claim older than STALE seconds without output is taken
over, so a crashed worker's task is rerun. A unit's output does not depend on which worker ran it (given the GPU type).

Usage: XL_MODEL=<model> python recover.py work [--main]   (recover.sh runs workers over all models)
       XL_MODEL=<model> python recover.py analyse
`work` runs every task of the model that is free now and exits 0 if the model has nothing left (with --main: only
once its analysis exists), 2 if it did some work, 3 if work is pending on other workers.
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from generate_pairs import GEN, KEY, LANGS, MODEL, MODELS, PAIRS, RES, TUNE, Model, flores, prompt, split  # noqa: E402
from tasks import MGSM  # noqa: E402

ARM = "recover"
BASE_ALPHAS = (0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16)
EXTEND = 4  # doublings added at once when the best dev alpha is the largest of the grid
MAX_ALPHA = 2 ** 20
HELD = tuple(p for p in PAIRS if p not in TUNE)
OUT = GEN / "recover"
CLAIMS = GEN / "claims" / "recover"
STALE = 15 * 60
# the existing test runs it is compared with (generate_pairs.py)
REFS = ("instruction", "instruction_steer_prompt", "instruction_steer")


def dev_path(alpha):
    return OUT / "dev" / f"{ARM}_a{alpha}_r1.jsonl"


_dev_means = {}


def dev_mean(alpha):
    """Mean dev success of alpha over the 11 tuning pairs."""
    if alpha not in _dev_means:
        from analyse import load, score  # noqa: E402  (../mgsm_steering/analyse.py; loads fastText)
        rows = load(dev_path(alpha))
        keys = [KEY[p] for p in TUNE]
        by = {k: [r for r in rows if KEY[(r["src"], r["tgt"])] == k] for k in keys}
        _dev_means[alpha] = sum(score(by[k])["success"] for k in keys) / len(keys)
    return _dev_means[alpha]


def best(grid):
    return min(grid, key=lambda a: (-round(dev_mean(a), 9), a))


def alphas():
    """The alpha grid so far: BASE_ALPHAS, extended while every dev unit of it exists and the best is the largest."""
    g = list(BASE_ALPHAS)
    while all(dev_path(a).exists() for a in g) and best(g) == g[-1] and g[-1] < MAX_ALPHA:
        g += [g[-1] * 2 ** k for k in range(1, EXTEND + 1)]
    return tuple(g)


def dev_units():
    items = [(r, s, t) for s, t in TUNE for r in split()["dev"]]
    return {dev_path(a).stem: (a, items) for a in alphas()}


def test_units():
    alpha = json.loads((OUT / "chosen.json").read_text())["alpha"]
    test = split()["test"]
    return {f"{ARM}_{KEY[p]}": (alpha, [(r, *p) for r in test]) for p in HELD}


def run_unit(M, mu, alpha, items):
    texts = [prompt("instruction", r, s, t) for r, s, t in items]
    res = M.generate(texts, vecs=[mu[t] - mu[s] for _, s, t in items], recover=dict(alpha=alpha, restore_norm=True))
    task = MGSM()
    return [dict(id=r["id"], src=s, tgt=t, lang=t, arm=ARM, layer="all", alpha=alpha, restore_norm=True,
                 prompt_only=False, answer=r["answer"], pred=task.parse(out), system=None, input=x, response=out,
                 n_tokens=k, capped=c)
            for (r, s, t), x, (out, k, c) in zip(items, texts, res)]


# ---------------------------------------------------------------- claims
class Claim:
    """An exclusive claim on a task, kept fresh by a heartbeat thread while held."""

    def __init__(self, path):
        self.path, self.stop = path, threading.Event()

    def take(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and time.time() - self.path.stat().st_mtime > STALE:
            try:  # a dead worker's claim: whoever renames it first takes the task over
                self.path.rename(self.path.with_name(f"{self.path.name}.stale.{uuid.uuid4().hex[:8]}"))
            except FileNotFoundError:
                return False
            print(f"taking over the stale claim {self.path.name}", flush=True)
        try:
            os.close(os.open(self.path, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            return False
        threading.Thread(target=self._beat, daemon=True).start()
        return True

    def _beat(self):
        while not self.stop.wait(60):
            self.path.touch()

    def release(self):
        self.stop.set()


def status(path, claim):
    """'done' (output exists), 'free' (unclaimed or stale) or 'running'."""
    if path.exists():
        return "done"
    if claim.exists() and time.time() - claim.stat().st_mtime <= STALE:
        return "running"
    return "free"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.rename(path)


# ---------------------------------------------------------------- tasks
def tasks():
    """[(task, output path, claim path)] that can run now, the number still running, and whether the model is done
    (its analysis exists)."""
    seq = [("vectors", OUT / "vectors.pt")]
    if (OUT / "vectors.pt").exists():
        seq += [(f"dev/{n}", OUT / "dev" / f"{n}.jsonl") for n in dev_units()]
        if all((OUT / "dev" / f"{n}.jsonl").exists() for n in dev_units()):
            seq.append(("select", OUT / "chosen.json"))
            if (OUT / "chosen.json").exists():
                seq += [(f"test/{n}", OUT / "test" / f"{n}.jsonl") for n in test_units()]
                if all((OUT / "test" / f"{n}.jsonl").exists() for n in test_units()):
                    seq.append(("analyse", RES / "recover.json"))
    st = [(t, o, CLAIMS / t.replace("/", "_"), status(o, CLAIMS / t.replace("/", "_"))) for t, o in seq]
    free = [(t, o, c) for t, o, c, s in st if s == "free"]
    running = sum(s == "running" for *_, s in st)
    done = seq[-1][0] == "analyse" and st[-1][3] == "done"
    return free, running, done


def work(main):
    """Run every free task of the model; see the module docstring for the exit code."""
    M, mu, did = None, None, False
    while True:
        free, running, done = tasks()
        if not free:
            break
        t, out, cpath = free[0]
        c = Claim(cpath)
        if t.startswith(("vectors", "dev/", "test/")) and M is None:
            M = Model(MODELS[MODEL]["hf"])  # loaded before claiming: loading can take minutes
        if not c.take():
            time.sleep(1)  # someone else took it; look again
            continue
        try:
            if t == "vectors":
                v = {L: M.mean_states(flores(L)).cpu() for L in LANGS}
                old = torch.load(GEN / "vectors.pt")  # generate_pairs.py's, at the grid layers: must agree
                diff = max((v[L][l] - old[L][l]).abs().max().item() for L in LANGS for l in old[LANGS[0]])
                print(f"vectors: max |diff| to generate_pairs.py's at the grid layers {diff:.2e}", flush=True)
                tmp = OUT / ".vectors.pt.tmp"
                OUT.mkdir(parents=True, exist_ok=True)
                torch.save(v, tmp)
                tmp.rename(out)
            elif t == "select":
                select()
            elif t == "analyse":
                analyse()
            else:
                stage, name = t.split("/")
                if mu is None:
                    mu = torch.load(OUT / "vectors.pt")
                alpha, items = (dev_units() if stage == "dev" else test_units())[name]
                t0 = time.time()
                rows = run_unit(M, mu, alpha, items)
                write_rows(out, rows)
                acc = sum(r["pred"] is not None and abs(r["pred"] - r["answer"]) < 1e-6 for r in rows) / len(rows)
                print(f"{MODEL} {t}: acc {acc:.2f} capped {sum(r['capped'] for r in rows)} "
                      f"({time.time() - t0:.0f} s) | {rows[0]['response'][:60]!r}", flush=True)
            did = True
        finally:
            c.release()
    if did:
        return 2
    free, running, done = tasks()
    if done:
        return 0
    # a helper leaves once the test units exist (the last GPU tasks) and are all taken: the worker that finishes
    # the last one runs the analysis, and the main job takes over any task whose worker dies
    if not main and not free and (OUT / "chosen.json").exists():
        return 0
    return 3


# ---------------------------------------------------------------- selection and analysis
def select():
    g = alphas()
    a = best(g)
    edge = a in (g[0], g[-1])
    chosen = dict(alpha=a, restore_norm=True, mean_success=dev_mean(a), at_grid_edge=edge)
    print(f"\n{ARM}: mean dev success over the tuning pairs")
    print("".join(f"{x:>9}" for x in g))
    print("".join(f"{dev_mean(x):>9.3f}" for x in g))
    print(f"chosen: alpha {a} ({dev_mean(a):.3f})" + (" AT THE GRID EDGE" if edge else ""))
    (OUT / "dev_grid.json").write_text(json.dumps([dict(alpha=x, mean_success=dev_mean(x)) for x in g], indent=1) + "\n")
    tmp = OUT / ".chosen.json.tmp"
    tmp.write_text(json.dumps(chosen, indent=1) + "\n")
    tmp.rename(OUT / "chosen.json")


def analyse():
    import numpy as np
    from agreement import kappa  # noqa: E402
    from analyse import correct, holm, load, mcnemar, score, success  # noqa: E402
    from native_tests import N_BOOT, SEED, ci, fmt  # noqa: E402

    T = GEN / "test"
    rows = {}
    for p in HELD:
        k = KEY[p]
        rows["native", p] = load(T / f"native_{p[1]}.jsonl")
        for a in REFS:
            rows[a, p] = load(T / f"{a}_{k}.jsonl")
        rows[ARM, p] = load(OUT / "test" / f"{ARM}_{k}.jsonl")
    chosen = json.loads((OUT / "chosen.json").read_text())
    assert all((r["alpha"], r["restore_norm"], r["prompt_only"]) == (chosen["alpha"], True, False)
               for p in HELD for r in rows[ARM, p]), "test runs not at the chosen setting"
    ids = [r["id"] for r in rows["instruction", HELD[0]]]
    assert all([r["id"] for r in rs] == ids for rs in rows.values())
    ALL = ("native",) + REFS + (ARM,)
    ok = {key: np.array([success(r) for r in rs], dtype=float) for key, rs in rows.items()}
    cor = {key: np.array([correct(r) for r in rs]) for key, rs in rows.items()}
    sc = {key: score(rs) for key, rs in rows.items()}
    n = len(ids)
    boot = np.random.default_rng(SEED).integers(0, n, (N_BOOT, n))
    item_mean = lambda a: np.mean([ok[a, p] for p in HELD], axis=0)  # per item, over the held-out pairs
    keys = [KEY[p] for p in HELD]
    out = {"held_out_pairs": keys, "chosen": chosen,
           "dev_grid": json.loads((OUT / "dev_grid.json").read_text()), "n_boot": N_BOOT, "seed": SEED,
           "means": {a: {c: float(np.mean([sc[a, p][c] for p in HELD])) for c in
                         ("success", "accuracy", "in_lang", "english", "capped")} | {"success_ci": ci(item_mean(a), boot)}
                     for a in ALL},
           "comparisons": {}}
    # (arm, reference): McNemar per held-out pair on success, Holm over the 99 pairs; mean difference with its CI
    for a, b in ((ARM, "instruction"), (ARM, "instruction_steer_prompt"), (ARM, "instruction_steer"), (ARM, "native")):
        per = {}
        for p in HELD:
            g, l, pv = mcnemar([bool(x) for x in ok[b, p]], [bool(x) for x in ok[a, p]])
            per[KEY[p]] = dict(gained=int(g), lost=int(l), p=pv)
        for d, q in zip(per.values(), holm([d["p"] for d in per.values()])):
            d["p_holm"] = q
        cnt = lambda k, s: sum(d[k] < 0.05 and (d["gained"] - d["lost"]) * s > 0 for d in per.values())
        out["comparisons"][f"{a}-{b}"] = dict(
            diff_ci=ci(item_mean(a) - item_mean(b), boot),
            better_worse={k: [cnt(k, 1), cnt(k, -1)] for k in ("p", "p_holm")}, per_pair=per)
    # error consistency with native (agreement.py): kappa on per-item correctness, mean over the held-out pairs
    idx = np.concatenate([np.arange(n)[None], boot])
    kap = {a: np.stack([kappa(cor[a, p][idx], cor["native", p][idx]) for p in HELD], 1) for a in REFS + (ARM,)}
    summ = lambda x: dict(mean=float(x[0].mean()), ci=[float(v) for v in np.percentile(x[1:].mean(1), [2.5, 97.5])])
    out["kappa_native"] = {a: summ(kap[a]) for a in kap}
    out["kappa_native_diff"] = {f"{ARM}-instruction_steer_prompt": summ(kap[ARM] - kap["instruction_steer_prompt"])}

    print(f"\n{MODEL}: {len(HELD)} held-out pairs, test (n = {n} per pair); chosen on dev: alpha {chosen['alpha']}")
    print(f"{'':26}{'success [95% CI]':>24}{'accuracy':>10}{'in T':>8}{'capped':>8}{'kappa native':>14}")
    for a in ALL:
        m = out["means"][a]
        kn = f"{out['kappa_native'][a]['mean']:.3f}" if a in out["kappa_native"] else ""
        print(f"{a:26}{fmt(m['success_ci']):>24}{m['accuracy']:>10.3f}{m['in_lang']:>8.3f}{m['capped']:>8.3f}{kn:>14}")
    print("\narm - reference: mean success difference [95% CI]; pairs better/worse, p < 0.05 (Holm)")
    for name, d in out["comparisons"].items():
        bw = d["better_worse"]
        print(f"{name:42}{fmt(d['diff_ci'], '+'):>26}   {bw['p'][0]}/{bw['p'][1]} ({bw['p_holm'][0]}/{bw['p_holm'][1]})")
    for name, d in out["kappa_native_diff"].items():
        print(f"kappa with native, {name}: {d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
    tmp = RES / ".recover.json.tmp"
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    tmp.rename(RES / "recover.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("work", "analyse"))
    ap.add_argument("--main", action="store_true", help="work: wait for the model's analysis before exiting 0")
    a = ap.parse_args()
    if a.stage == "analyse":
        analyse()
    else:
        sys.exit(work(a.main))


if __name__ == "__main__":
    main()
