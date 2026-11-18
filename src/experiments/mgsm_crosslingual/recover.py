"""ReCoVeR (Sterz, Schmidt, Glavaš, Vulić, Findings of EMNLP 2025; code github.com/hSterz/recover @ 4c6eb4d) as a
baseline on cross-lingual MGSM: its training-free ("naive", unsupervised) steering, reimplemented in
Model.generate (../mgsm_steering/generate.py, recover=...; checked against their code by recover_check.py), on top
of the instruction prompt, against our instruction + prompt-only steering at one layer. ReCoVeR+ (trained) is not run.

ReCoVeR adds alpha * (r_T - r_S)(j) / |(r_T - r_S)(j)| to the output of every block j, at every position but the first
token of the prompt (the chat template included) and at every generated token, with an optional norm restore
(each output rescaled to the norm of the unsteered one). r_L(j) = mu_L(j) - (the mean over languages), mu_L(j) the
mean hidden state j over the tokens (the first excluded) of FLORES sentences in L; the centring cancels in r_T - r_S,
so the vector is mu_T(j) - mu_S(j), from the same 997 FLORES-200 dev sentences and Model.mean_states as ours. As in
their code, the vector of the last block is the mean of the model's last hidden state, which Hugging Face returns
after the final norm.

Arms, both on the instruction prompt (the S question + "Answer in <T>." in S, the `instruction` arm):
  recover          ReCoVeR as published: all blocks, every token
  recover_prompt   the same, but the generated tokens are not steered (prompt only, as ours)
Each arm's alpha (ALPHAS, in units of the normalised vector, covering their 0.2 and 1) and norm restore (off/on)
are chosen on the dev items of the 11 tuning pairs by the rule of analyse_pairs.py (highest mean success; ties to
the smaller alpha, then restore off), then each is run on the test items of the 99 held-out pairs. Prompts,
decoding (greedy, at most 2048 new tokens), batching, split and scoring are those of generate_pairs.py.

Tasks per model, shared by any number of workers through claim files (one per task, under
results/<model>/generations/claims/recover/), in order:
  vectors   mu_L(j) for every hidden state j and the 11 languages -> generations/recover/vectors.pt
  dev       one unit per (arm, alpha, restore) over the 11 tuning pairs -> generations/recover/dev/<unit>.jsonl
  select    -> generations/recover/chosen.json
  test      one unit per (arm, held-out pair) -> generations/recover/test/<arm>_<S>-<T>.jsonl
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

ARMS = ("recover", "recover_prompt")
ALPHAS = (0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16)
RESTORE = (False, True)
HELD = tuple(p for p in PAIRS if p not in TUNE)
OUT = GEN / "recover"
CLAIMS = GEN / "claims" / "recover"
STALE = 15 * 60
# the existing test runs it is compared with (generate_pairs.py)
REFS = ("instruction", "instruction_steer_prompt", "instruction_steer")


def unit_name(arm, alpha, restore):
    return f"{arm}_a{alpha}_r{int(restore)}"


def dev_units():
    items = [(r, s, t) for s, t in TUNE for r in split()["dev"]]
    return {unit_name(arm, a, rn): (arm, a, rn, items) for arm in ARMS for a in ALPHAS for rn in RESTORE}


def test_units():
    chosen = json.loads((OUT / "chosen.json").read_text())
    test = split()["test"]
    return {f"{arm}_{KEY[p]}": (arm, chosen[arm]["alpha"], chosen[arm]["restore_norm"], [(r, *p) for r in test])
            for arm in ARMS for p in HELD}


def run_unit(M, mu, arm, alpha, restore, items):
    texts = [prompt("instruction", r, s, t) for r, s, t in items]
    po = arm == "recover_prompt"
    res = M.generate(texts, vecs=[mu[t] - mu[s] for _, s, t in items], prompt_only=po,
                     recover=dict(alpha=alpha, restore_norm=restore))
    task = MGSM()
    return [dict(id=r["id"], src=s, tgt=t, lang=t, arm=arm, layer="all", alpha=alpha, restore_norm=restore,
                 prompt_only=po, answer=r["answer"], pred=task.parse(out), system=None, input=x, response=out,
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
                arm, alpha, restore, items = (dev_units() if stage == "dev" else test_units())[name]
                t0 = time.time()
                rows = run_unit(M, mu, arm, alpha, restore, items)
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
    from analyse import load, score  # noqa: E402  (../mgsm_steering/analyse.py; loads fastText)

    keys = [KEY[p] for p in TUNE]
    chosen, grid = {}, {}
    for arm in ARMS:
        means = {}
        for a in ALPHAS:
            for rn in RESTORE:
                rows = load(OUT / "dev" / f"{unit_name(arm, a, rn)}.jsonl")
                by = {k: [r for r in rows if KEY[(r["src"], r["tgt"])] == k] for k in keys}
                means[(a, rn)] = sum(score(by[k])["success"] for k in keys) / len(keys)
        (a, rn) = min(means, key=lambda k: (-round(means[k], 9), k[0], k[1]))
        edge = a in (ALPHAS[0], ALPHAS[-1])
        chosen[arm] = dict(alpha=a, restore_norm=rn, mean_success=means[(a, rn)], at_grid_edge=edge)
        grid[arm] = [dict(alpha=a, restore_norm=rn, mean_success=m) for (a, rn), m in sorted(means.items())]
        print(f"\n{arm}: mean dev success over the tuning pairs (rows: restore_norm, columns: alpha)")
        print(f"{'':8}" + "".join(f"{x:>7}" for x in ALPHAS))
        for r in RESTORE:
            print(f"{str(r):8}" + "".join(f"{means[(x, r)]:>7.3f}" for x in ALPHAS))
        print(f"chosen: alpha {a}, restore_norm {rn} ({means[(a, rn)]:.3f})" + (" AT THE GRID EDGE" if edge else ""))
    (OUT / "dev_grid.json").write_text(json.dumps(grid, indent=1) + "\n")
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
        for a in ARMS:
            rows[a, p] = load(OUT / "test" / f"{a}_{k}.jsonl")
    ids = [r["id"] for r in rows["instruction", HELD[0]]]
    assert all([r["id"] for r in rs] == ids for rs in rows.values())
    ALL = ("native",) + REFS + ARMS
    ok = {key: np.array([success(r) for r in rs], dtype=float) for key, rs in rows.items()}
    cor = {key: np.array([correct(r) for r in rs]) for key, rs in rows.items()}
    sc = {key: score(rs) for key, rs in rows.items()}
    n = len(ids)
    boot = np.random.default_rng(SEED).integers(0, n, (N_BOOT, n))
    item_mean = lambda a: np.mean([ok[a, p] for p in HELD], axis=0)  # per item, over the held-out pairs
    keys = [KEY[p] for p in HELD]
    out = {"held_out_pairs": keys, "chosen": json.loads((OUT / "chosen.json").read_text()),
           "dev_grid": json.loads((OUT / "dev_grid.json").read_text()), "n_boot": N_BOOT, "seed": SEED,
           "means": {a: {c: float(np.mean([sc[a, p][c] for p in HELD])) for c in
                         ("success", "accuracy", "in_lang", "english", "capped")} | {"success_ci": ci(item_mean(a), boot)}
                     for a in ALL},
           "comparisons": {}}
    # (arm, reference): McNemar per held-out pair on success, Holm over the 99 pairs; mean difference with its CI
    for a, b in (("recover", "instruction"), ("recover_prompt", "instruction"),
                 ("recover", "instruction_steer_prompt"), ("recover_prompt", "instruction_steer_prompt"),
                 ("recover", "instruction_steer"), ("recover", "recover_prompt"),
                 ("recover", "native"), ("recover_prompt", "native")):
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
    kap = {a: np.stack([kappa(cor[a, p][idx], cor["native", p][idx]) for p in HELD], 1) for a in REFS + ARMS}
    summ = lambda x: dict(mean=float(x[0].mean()), ci=[float(v) for v in np.percentile(x[1:].mean(1), [2.5, 97.5])])
    out["kappa_native"] = {a: summ(kap[a]) for a in kap}
    out["kappa_native_diff"] = {f"{a}-instruction_steer_prompt": summ(kap[a] - kap["instruction_steer_prompt"])
                                for a in ARMS}

    print(f"\n{MODEL}: {len(HELD)} held-out pairs, test (n = {n} per pair); chosen on dev: " +
          ", ".join(f"{a} alpha {out['chosen'][a]['alpha']} restore {out['chosen'][a]['restore_norm']}" for a in ARMS))
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
