"""How often an arm fails outright: the number of held-out pairs whose test success is below 0.10 and below 0.30.

Reads the per-pair test success from results/<model>/summary.json (written by analyse_pairs.py), for every arm tested
there (the instruction variant only if it won on dev); the 11 tuning pairs are left out, as in the held-out columns there.

Usage: XL_MODEL=<model> python src/experiments/mgsm_crosslingual/robustness.py   # -> results/<model>/robustness.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from generate_pairs import RES  # noqa: E402

THRESHOLDS = (0.10, 0.30)


def main():
    s = json.loads((RES / "summary.json").read_text())
    held = [k for k in s["test"] if k not in s["tuning_pairs"]]
    arms = list(s["means"]["held_out"])
    out = {"n_pairs": len(held), "thresholds": THRESHOLDS,
           "below": {a: {str(x): sum(s["test"][k][a]["success"] < x for k in held) for x in THRESHOLDS}
                     for a in arms}}
    print(f"held-out pairs ({len(held)}) with success below " + ", ".join(map(str, THRESHOLDS)))
    for a in arms:
        print(f"{a:26}" + "".join(f"{out['below'][a][str(x)]:>6}" for x in THRESHOLDS))
    (RES / "robustness.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
