"""Two more prompt baselines against the default instruction, on test: the instruction written in T
(instruction_target) and the instruction before the question (instruction_first); see generate_pairs.py.

They have no settings, so they are not selected on dev: each is compared directly with the instruction, by the
measures of analyse_pairs.py (an exact McNemar test on per-item success within each pair; means over the tuning,
held-out and all pairs; held-out pairs significantly better / worse, uncorrected and Holm-corrected), and with the failure counts of
robustness.py (held-out pairs with success below 0.10 and 0.30). The instruction and
instruction + prompt-only steering are listed for reference, from the same test files. Waits until every unit of
the prompts stage exists (helper jobs may still be running some).

Usage: XL_MODEL=<model> python src/experiments/mgsm_crosslingual/prompt_variants.py   # -> results/<model>/prompt_variants.json
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from analyse import load, mcnemar, score, success  # noqa: E402
from analyse_pairs import COLS, HELD, better_worse, correct  # noqa: E402
from generate_pairs import GEN, KEY, PAIRS, PROMPT_ARMS, RES, TUNE, units  # noqa: E402
from robustness import THRESHOLDS  # noqa: E402

ARMS = ("instruction",) + PROMPT_ARMS + ("instruction_steer_prompt",)


def main():
    T = GEN / "test"
    while missing := [n for n in units("prompts") if not (T / f"{n}.jsonl").exists()]:  # helper jobs may still run
        print(f"waiting for {len(missing)} units", flush=True)
        time.sleep(30)
    test = {}
    for p in PAIRS:
        k = KEY[p]
        rows = {arm: load(T / f"{arm}_{k}.jsonl") for arm in ARMS}
        base = [success(r) for r in rows["instruction"]]
        test[k] = {}
        for arm in ARMS:
            d = test[k][arm] = score(rows[arm])
            if arm != "instruction":
                assert [r["id"] for r in rows[arm]] == [r["id"] for r in rows["instruction"]]
                n01, n10, pv = mcnemar(base, [success(r) for r in rows[arm]])
                d.update(gained=n01, lost=n10, p=pv)
    groups = {"tuning": TUNE, "held_out": HELD, "all": PAIRS}
    means = {g: {arm: {c: sum(test[KEY[p]][arm][c] for p in ps) / len(ps) for c in COLS} for arm in ARMS}
             for g, ps in groups.items()}
    held = [test[KEY[p]] for p in HELD]
    for arm in ARMS[1:]:
        correct([d[arm] for d in held])
    bw = {arm: better_worse([d[arm] for d in held]) for arm in ARMS[1:]}
    below = {arm: {str(x): sum(d[arm]["success"] < x for d in held) for x in THRESHOLDS} for arm in ARMS}

    print(f"== mean over the {len(HELD)} held-out pairs")
    print(f"{'':26}" + "".join(f"{c:>9}" for c in COLS) + "   better / worse (Holm)   " +
          "   ".join(f"< {x}" for x in THRESHOLDS))
    for arm in ARMS:
        sig = (f"{bw[arm]['p'][0]:>7} / {bw[arm]['p'][1]:<3} ({bw[arm]['p_holm'][0]} / {bw[arm]['p_holm'][1]})".ljust(22)
               if arm in bw else " " * 22)
        print(f"{arm:26}" + "".join(f"{means['held_out'][arm][c]:>9.3f}" for c in COLS) + f"   {sig}   " +
              "".join(f"{below[arm][str(x)]:>7}" for x in THRESHOLDS))
    (RES / "prompt_variants.json").write_text(json.dumps(
        dict(arms=ARMS, means=means, better_worse=bw, below=below, test=test), indent=1) + "\n")


if __name__ == "__main__":
    main()
