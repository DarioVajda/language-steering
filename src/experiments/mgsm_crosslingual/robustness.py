"""How often an arm fails outright: the number of held-out pairs whose test success is below 0.10 and below 0.30.

Reads the per-pair test success from results/summary.json (written by analyse_pairs.py); the 11 tuning pairs are
left out, as in the held-out columns there.

Usage: python src/experiments/mgsm_crosslingual/robustness.py   # -> results/robustness.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("source", "native", "instruction", "steer", "instruction_steer", "steer_prompt", "instruction_steer_prompt")
THRESHOLDS = (0.10, 0.30)


def main():
    s = json.loads((HERE / "results/summary.json").read_text())
    held = [k for k in s["test"] if k not in s["tuning_pairs"]]
    out = {"n_pairs": len(held), "thresholds": THRESHOLDS,
           "below": {a: {str(x): sum(s["test"][k][a]["success"] < x for k in held) for x in THRESHOLDS}
                     for a in ARMS}}
    print(f"held-out pairs ({len(held)}) with success below " + ", ".join(map(str, THRESHOLDS)))
    for a in ARMS:
        print(f"{a:26}" + "".join(f"{out['below'][a][str(x)]:>6}" for x in THRESHOLDS))
    (HERE / "results/robustness.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
