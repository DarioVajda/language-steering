"""ICL methods: how the few-shot prompt is built and how (if at all) the model is steered.

A Method subclass sets
  * `demo_langs(target, k, rng)`: the language of each demonstration;
  * `steering(config, target)`: a steering.Steering for a hyperparameter config, or None;
  * `grid`: hyperparameter -> candidate values. Empty means nothing to tune (see run.py);
  * `fallback`: a config tried alongside the grid that wins ties, e.g. "no steering", so a method
    falls back to the unsteered baseline when no grid config beats it on val (Kirtane & Huang).
and inherits `prompt` (build one prompt), `forward` / `generate` (the two kinds of forward pass)
and `run` (score a list of items).

The demonstration items depend only on (seed, query item), never on the method, so all methods
see the same demonstrations and their results are paired.

Baselines (Kirtane & Huang, 2026):
  en      English demonstrations
  mfs     multilingual few-shot: each demonstration in a random language other than the target
  oracle  target-language demonstrations
  steer   English demonstrations, h <- h + alpha * v_(en->T) at one layer (on chosen positions)

Controls:
  random  as steer on the demonstrations, but v is a random direction with the norm of v_(en->T)
  wrong   as steer on the demonstrations, but v is v_(en->L') of another language L', with the
          norm of v_(en->T)

New methods:
  mfs_steer  MFS demonstrations, each shifted by v_(L_p->T) of its own language L_p

and its controls, as mfs_steer but with demonstrations in L_p shifted by
  mfs_random   a random direction per L_p, with the norm of v_(L_p->T)
  mfs_wrong    v_(L_p->L') towards another language L', with the norm of v_(L_p->T)
  mfs_uniform  v_(en->T), whatever L_p
"""

import itertools

import torch

from model import Prompt, Segment
from steering import AddLanguageVector, AddVector
from tasks import K_SHOT
from vectors import item_rng


class Method:
    name = None
    grid = {}
    fallback = None
    uses_vectors = False

    def __init__(self, task, runner, vectors=None, k=K_SHOT, batch_size=16):
        self.task, self.runner, self.vectors, self.k, self.batch_size = task, runner, vectors, k, batch_size
        self.pool = task.demo_pool()

    def configs(self):
        """All configs to try on val: the fallback first, then the grid."""
        keys = list(self.grid)
        grid = [dict(zip(keys, values)) for values in itertools.product(*self.grid.values())]
        return ([self.fallback] if self.fallback is not None else []) + grid

    def demo_langs(self, target, k, rng):
        raise NotImplementedError

    def steering(self, config, target):
        return None

    def prompt(self, item, target, seed):
        demos = [self.pool[j] for j in item_rng(seed, item.id).choice(len(self.pool), self.k, replace=False)]
        langs = self.demo_langs(target, self.k, item_rng(seed, item.id, "langs"))
        return Prompt((Segment("instruction", self.task.instruction),
                       *(Segment("demo", self.task.format_demo(d, L), L) for d, L in zip(demos, langs)),
                       Segment("query", self.task.format_query(item, target), target)), target)

    def forward(self, prompts, config, target, token_ids=None):
        return self.runner.next_token_logprobs(prompts, self.task.langs, self.steering(config, target),
                                               token_ids, self.batch_size)

    def generate(self, prompts, config, target):
        return self.runner.generate(prompts, self.task.langs, self.steering(config, target),
                                    self.task.max_new_tokens, self.task.stop, self.batch_size)

    def run(self, items, target, config, seed, mode=None):
        """Score `items` in language `target`: one record per item.

        mode "generate": greedy generation, parsed by the task (the protocol of the paper);
        mode "forward":  one forward pass, prediction = most likely label by its first token
                         (closed label sets only); the label log-probs are kept."""
        mode = mode or self.task.mode
        prompts = [self.prompt(it, target, seed) for it in items]
        if mode == "generate":
            responses = self.generate(prompts, config, target)
            preds = [self.task.parse(r) for r in responses]
            extra = [{"response": r} for r in responses]
        elif mode == "forward":
            assert self.task.labels, f"{self.task.name} has no closed label set"
            first = [self.runner.tokenizer(l, add_special_tokens=False).input_ids[0] for l in self.task.labels]
            assert len(set(first)) == len(first), "labels share their first token"
            lp = self.forward(prompts, config, target, first)
            preds = [self.task.labels[i] for i in lp.argmax(1).tolist()]
            extra = [{"logprobs": dict(zip(self.task.labels, row))} for row in lp.tolist()]
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return [{"id": it.id, "lang": target, "seed": seed, "pred": p, "answer": it.answer,
                 "correct": self.task.is_correct(p, it), **e} for it, p, e in zip(items, preds, extra)]


class EnglishFewShot(Method):
    name = "en"

    def demo_langs(self, target, k, rng):
        return ["en"] * k


class MultilingualFewShot(Method):
    """Each demonstration in a language drawn uniformly (with replacement) from the benchmark's
    languages other than the target, English included (cf. Tu et al., 2025)."""
    name = "mfs"

    def demo_langs(self, target, k, rng):
        others = [L for L in self.task.langs if L != target]
        return [others[j] for j in rng.integers(len(others), size=k)]


class Oracle(Method):
    name = "oracle"

    def demo_langs(self, target, k, rng):
        return [target] * k


class ConstantSteering(EnglishFewShot):
    """Kirtane & Huang (2026): English demos, h <- h + alpha * v_(en->T) at one layer.
    Grid as in the paper; positions "demos" is their on_fewshot, "boundary" after_fewshot,
    "query" on_question and "prompt" entire."""
    name = "steer"
    grid = {"layer": [5, 10, 15, 20, 25, 30], "alpha": [0.5, 1.0, 2.0, 3.0],
            "positions": ["demos", "boundary", "query", "prompt"]}
    fallback = {"alpha": 0.0}
    uses_vectors = True

    def steering(self, config, target):
        if not config.get("alpha"):
            return None
        v = self.vectors.direction("en", target, config["layer"])
        return AddVector(config["layer"], v, config["alpha"], config["positions"])


class RandomSteering(ConstantSteering):
    """Control for steer: a random Gaussian direction per (draw, layer), shared by all target
    languages and rescaled to the norm of v_(en->T) at that layer. Each draw is tuned on its own,
    over the same layer x alpha grid as steer on the demonstrations (run.py --grid draw=D)."""
    name = "random"
    grid = {"layer": [5, 10, 15, 20, 25, 30], "alpha": [0.5, 1.0, 2.0, 3.0], "positions": ["demos"], "draw": [0]}

    def steering(self, config, target):
        if not config.get("alpha"):
            return None
        v = self.vectors.direction("en", target, config["layer"])
        u = torch.from_numpy(item_rng(config["draw"], "random_direction", config["layer"]).standard_normal(len(v)))
        u = (u / u.norm() * v.norm()).float()
        return AddVector(config["layer"], u, config["alpha"], config["positions"])


class WrongLanguageSteering(ConstantSteering):
    """Control for steer: the vector of another language L', rescaled to the norm of v_(en->T).
    L' is drawn per (draw, target) from the benchmark's languages other than en and T, and each
    draw is tuned on its own (run.py --grid draw=D). A config with "steer_lang" uses that language
    for every target instead (the transfer matrix; steer_lang = T reproduces steer)."""
    name = "wrong"
    grid = {"layer": [5, 10, 15, 20, 25, 30], "alpha": [0.5, 1.0, 2.0, 3.0], "positions": ["demos"], "draw": [0]}

    def steer_lang(self, config, target):
        if "steer_lang" in config:
            return config["steer_lang"]
        others = [L for L in self.task.langs if L not in ("en", target)]
        return others[item_rng(config["draw"], "wrong_language", target).integers(len(others))]

    def steering(self, config, target):
        if not config.get("alpha"):
            return None
        v = self.vectors.direction("en", target, config["layer"])
        u = self.vectors.direction("en", self.steer_lang(config, target), config["layer"])
        return AddVector(config["layer"], u / u.norm() * v.norm(), config["alpha"], config["positions"])


class MultilingualSteering(MultilingualFewShot):
    """MFS + steering: MFS demonstrations, and each demonstration's tokens shifted towards the target
    by the vector of their own language, h <- h + alpha * v_(L_p->T) at one layer, where L_p is the
    language of demonstration p (en demonstrations get v_(en->T), as in steer). The fallback
    alpha = 0 is plain MFS. Subclasses (the controls) change only `source_vectors`."""
    name = "mfs_steer"
    grid = {"layer": [5, 10, 15, 20, 25, 30], "alpha": [0.5, 1.0, 2.0, 3.0], "positions": ["demos"]}
    fallback = {"alpha": 0.0}
    uses_vectors = True

    def source_vectors(self, config, target):
        """Row i: the vector added to demonstrations in language task.langs[i] (the language
        indexing of the encoded batch), here v_(L_i->T)."""
        return torch.stack([self.vectors.direction(L, target, config["layer"]) for L in self.task.langs])

    def steering(self, config, target):
        if not config.get("alpha"):
            return None
        return AddLanguageVector(config["layer"], self.source_vectors(config, target), config["alpha"],
                                 config["positions"])


def _rescale(U, V):
    """Rows of U rescaled to the norms of the rows of V; zero rows stay zero."""
    n = U.norm(dim=1, keepdim=True)
    return torch.where(n > 0, U / n.clamp(min=1e-12) * V.norm(dim=1, keepdim=True), torch.zeros_like(U))


class MultilingualRandomSteering(MultilingualSteering):
    """Control for mfs_steer: demonstrations in language L_p get a random Gaussian direction of their
    own, per (draw, layer, L_p) and shared by all targets, rescaled to the norm of v_(L_p->T).
    Each draw is tuned on its own (run.py --grid draw=D)."""
    name = "mfs_random"
    grid = {**MultilingualSteering.grid, "draw": [0]}

    def source_vectors(self, config, target):
        V = super().source_vectors(config, target)
        U = torch.stack([torch.from_numpy(item_rng(config["draw"], "mfs_random_direction", config["layer"], L)
                                          .standard_normal(V.shape[1])).float() for L in self.task.langs])
        return _rescale(U, V)


class MultilingualWrongSteering(MultilingualSteering):
    """Control for mfs_steer: demonstrations in language L_p get v_(L_p->L') towards another language
    L' instead of T, rescaled to the norm of v_(L_p->T); demonstrations already in L' are not
    steered (v_(L'->L') = 0). L' is drawn per (draw, target) from the languages other than en and
    T, as in wrong (the same L' for the same draw), and each draw is tuned on its own (run.py
    --grid draw=D). A config with "steer_lang" uses that L' for every target instead (the transfer
    matrix; steer_lang = T reproduces mfs_steer)."""
    name = "mfs_wrong"
    grid = {**MultilingualSteering.grid, "draw": [0]}
    steer_lang = WrongLanguageSteering.steer_lang

    def source_vectors(self, config, target):
        V = super().source_vectors(config, target)
        Lw = self.steer_lang(config, target)
        U = torch.stack([self.vectors.direction(L, Lw, config["layer"]) for L in self.task.langs])
        return _rescale(U, V)


class MultilingualUniformSteering(MultilingualSteering):
    """Control for mfs_steer: every demonstration gets v_(en->T), whatever its language, i.e. steer's
    vector on MFS demonstrations. Tests whether the per-demonstration vector matters."""
    name = "mfs_uniform"

    def source_vectors(self, config, target):
        v = self.vectors.direction("en", target, config["layer"])
        return v.expand(len(self.task.langs), -1).clone()


METHODS = {m.name: m for m in (EnglishFewShot, MultilingualFewShot, Oracle, ConstantSteering, RandomSteering,
                               WrongLanguageSteering, MultilingualSteering, MultilingualRandomSteering,
                               MultilingualWrongSteering, MultilingualUniformSteering)}
