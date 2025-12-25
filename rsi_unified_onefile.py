#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSI Unified One-File (Runnable)
===============================

This is a **single-file** runnable consolidation inspired by the user's five variants:
- an "all-in-one" launcher wrapper
- SCIG/SCIG+ style closed-loop symbolic regression with contracts and rollback
- an Ouroboros-style meta-loop that tunes hyperparameters and can emit a patched copy
- an L7-style lightweight atomic program-synthesis loop (NumPy optional)

Design choice:
- This file does NOT attempt to preserve every line/behavior of each 5k+ line variant.
- Instead, it **deduplicates** the overlapping mechanisms into one coherent, runnable system:
  * DSL (expression trees) + mutation operators
  * evaluation with contracts + holdout + stress gating
  * accept/rollback archive (JSONL + snapshots)
  * optional meta-tuning (Ouroboros)
  * optional NumPy-based atomic search (L7)

No external dependencies except **optional NumPy** for --engine l7.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np  # optional
except Exception:
    np = None


# =============================================================================
# Common utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


# =============================================================================
# Engine 1: SCIG+ (Symbolic Regression RSI loop)
# =============================================================================

# ---- DSL: expression trees ----

class Node:
    def eval(self, x: float) -> float:
        raise NotImplementedError
    def clone(self) -> "Node":
        raise NotImplementedError
    def size(self) -> int:
        raise NotImplementedError
    def walk(self) -> List["Node"]:
        raise NotImplementedError
    def replace_child(self, old: "Node", new: "Node") -> bool:
        return False
    def to_str(self) -> str:
        raise NotImplementedError

@dataclass
class Var(Node):
    def eval(self, x: float) -> float:
        return x
    def clone(self) -> "Node":
        return Var()
    def size(self) -> int:
        return 1
    def walk(self) -> List["Node"]:
        return [self]
    def to_str(self) -> str:
        return "x"

@dataclass
class Const(Node):
    c: float
    def eval(self, x: float) -> float:
        return float(self.c)
    def clone(self) -> "Node":
        return Const(float(self.c))
    def size(self) -> int:
        return 1
    def walk(self) -> List["Node"]:
        return [self]
    def to_str(self) -> str:
        return f"{float(self.c):.6g}"

@dataclass
class Unary(Node):
    op: str
    a: Node
    def eval(self, x: float) -> float:
        v = self.a.eval(x)
        try:
            if self.op == "neg":  return -v
            if self.op == "sin":  return math.sin(v)
            if self.op == "cos":  return math.cos(v)
            if self.op == "tanh": return math.tanh(v)
            if self.op == "abs":  return abs(v)
            if self.op == "log1p": return math.log1p(max(v, -0.999999))
            if self.op == "sqrt":  return math.sqrt(max(0.0, v))
        except Exception:
            return float("nan")
        raise ValueError(f"Unknown unary op: {self.op}")
    def clone(self) -> "Node":
        return Unary(self.op, self.a.clone())
    def size(self) -> int:
        return 1 + self.a.size()
    def walk(self) -> List["Node"]:
        return [self] + self.a.walk()
    def replace_child(self, old: Node, new: Node) -> bool:
        if self.a is old:
            self.a = new
            return True
        return self.a.replace_child(old, new)
    def to_str(self) -> str:
        return f"{self.op}({self.a.to_str()})"

@dataclass
class Binary(Node):
    op: str
    a: Node
    b: Node
    def eval(self, x: float) -> float:
        va = self.a.eval(x)
        vb = self.b.eval(x)
        try:
            if self.op == "add": return va + vb
            if self.op == "sub": return va - vb
            if self.op == "mul": return va * vb
            if self.op == "div":
                den = vb if abs(vb) > 1e-9 else (1e-9 if vb >= 0 else -1e-9)
                return va / den
            if self.op == "max": return max(va, vb)
            if self.op == "min": return min(va, vb)
            if self.op == "pow2": return va * va + vb * vb
        except Exception:
            return float("nan")
        raise ValueError(f"Unknown binary op: {self.op}")
    def clone(self) -> "Node":
        return Binary(self.op, self.a.clone(), self.b.clone())
    def size(self) -> int:
        return 1 + self.a.size() + self.b.size()
    def walk(self) -> List["Node"]:
        return [self] + self.a.walk() + self.b.walk()
    def replace_child(self, old: Node, new: Node) -> bool:
        if self.a is old:
            self.a = new
            return True
        if self.b is old:
            self.b = new
            return True
        return self.a.replace_child(old, new) or self.b.replace_child(old, new)
    def to_str(self) -> str:
        return f"{self.op}({self.a.to_str()}, {self.b.to_str()})"


UNARY_OPS = ["neg", "sin", "cos", "tanh", "abs", "log1p", "sqrt"]
BINARY_OPS = ["add", "sub", "mul", "div", "max", "min", "pow2"]

def random_leaf() -> Node:
    return Var() if random.random() < 0.55 else Const(random.uniform(-2.0, 2.0))

def random_tree(max_depth: int = 4) -> Node:
    if max_depth <= 0:
        return random_leaf()
    r = random.random()
    if r < 0.33:
        return random_leaf()
    if r < 0.63:
        return Unary(random.choice(UNARY_OPS), random_tree(max_depth - 1))
    return Binary(random.choice(BINARY_OPS), random_tree(max_depth - 1), random_tree(max_depth - 1))

def simplify(node: Node) -> Node:
    if isinstance(node, Unary):
        node.a = simplify(node.a)
        if node.op == "neg" and isinstance(node.a, Unary) and node.a.op == "neg":
            return node.a.a
        return node
    if isinstance(node, Binary):
        node.a = simplify(node.a)
        node.b = simplify(node.b)
        if isinstance(node.a, Const) and isinstance(node.b, Const):
            try:
                v = node.eval(0.0)
                if is_finite(v):
                    return Const(v)
            except Exception:
                pass
        if node.op == "mul":
            if isinstance(node.a, Const) and abs(node.a.c - 1.0) < 1e-9:
                return node.b
            if isinstance(node.b, Const) and abs(node.b.c - 1.0) < 1e-9:
                return node.a
            if isinstance(node.a, Const) and abs(node.a.c) < 1e-9:
                return Const(0.0)
            if isinstance(node.b, Const) and abs(node.b.c) < 1e-9:
                return Const(0.0)
        if node.op == "add":
            if isinstance(node.a, Const) and abs(node.a.c) < 1e-9:
                return node.b
            if isinstance(node.b, Const) and abs(node.b.c) < 1e-9:
                return node.a
        return node
    return node


# ---- contracts ----

@dataclass
class Contract:
    name: str
    check: Callable[[Callable[[float], float], List[float]], bool]

def contract_finite_and_bounded(bound: float = 1e6) -> Contract:
    def _check(fn: Callable[[float], float], xs: List[float]) -> bool:
        try:
            for x in xs:
                y = fn(x)
                if not is_finite(y):
                    return False
                if abs(float(y)) > bound:
                    return False
            return True
        except Exception:
            return False
    return Contract(name=f"finite_and_bounded({bound})", check=_check)

def contract_lipschitz_soft(max_slope: float = 1e4) -> Contract:
    def _check(fn: Callable[[float], float], xs: List[float]) -> bool:
        try:
            xs_sorted = sorted(xs)
            ys = [fn(x) for x in xs_sorted]
            for i in range(1, len(xs_sorted)):
                dx = xs_sorted[i] - xs_sorted[i - 1]
                if dx < 1e-9:
                    continue
                dy = abs(float(ys[i]) - float(ys[i - 1]))
                if (dy / dx) > max_slope:
                    return False
            return True
        except Exception:
            return False
    return Contract(name=f"lipschitz_soft({max_slope})", check=_check)

def contract_smooth_probe(max_local_slope: float = 2e5, dx: float = 1e-4) -> Contract:
    def _check(fn: Callable[[float], float], xs: List[float]) -> bool:
        try:
            for x in xs:
                y1 = fn(x)
                y2 = fn(x + dx)
                if not (is_finite(y1) and is_finite(y2)):
                    return False
                if abs(float(y2) - float(y1)) / dx > max_local_slope:
                    return False
            return True
        except Exception:
            return False
    return Contract(name=f"smooth_probe(max={max_local_slope},dx={dx})", check=_check)


# ---- test forge ----

@dataclass
class TestForgePlus:
    domain: Tuple[float, float] = (-3.0, 3.0)
    base_n: int = 64
    adversarial_n: int = 64
    boundary_n: int = 24
    regression_n: int = 32
    focus_strength: float = 0.50
    regression_bank: List[float] = field(default_factory=list)
    bank_max: int = 256

    def sample_base(self) -> List[float]:
        lo, hi = self.domain
        return [random.uniform(lo, hi) for _ in range(self.base_n)]

    def sample_boundary(self) -> List[float]:
        lo, hi = self.domain
        xs: List[float] = []
        for _ in range(self.boundary_n):
            if random.random() < 0.5:
                xs.append(lo + abs(random.gauss(0, 0.08)) * (hi - lo))
            else:
                xs.append(hi - abs(random.gauss(0, 0.08)) * (hi - lo))
        return xs

    def sample_regression(self) -> List[float]:
        if not self.regression_bank:
            return []
        k = min(self.regression_n, len(self.regression_bank))
        return random.sample(self.regression_bank, k=k)

    def add_regression_points(self, xs: List[float]) -> None:
        for x in xs:
            if is_finite(x):
                self.regression_bank.append(float(x))
        if len(self.regression_bank) > self.bank_max:
            self.regression_bank = self.regression_bank[-self.bank_max:]

    def sample_adversarial(self, fns: List[Callable[[float], float]]) -> List[float]:
        lo, hi = self.domain
        pool = [random.uniform(lo, hi) for _ in range(self.adversarial_n * 6)]
        scored: List[Tuple[float, float]] = []
        for x in pool:
            ys: List[float] = []
            valid: List[float] = []
            for fn in fns:
                try:
                    y = fn(x)
                    ys.append(y)
                    if is_finite(y):
                        valid.append(float(y))
                except Exception:
                    ys.append(float("nan"))
            if any(not is_finite(y) for y in ys):
                score = 1e9
            elif len(valid) > 1:
                m = sum(valid) / len(valid)
                score = sum((v - m) ** 2 for v in valid) / len(valid)
            else:
                score = 0.0
            scored.append((score, x))
        scored.sort(reverse=True, key=lambda t: t[0])
        top = [x for _, x in scored[: self.adversarial_n]]
        mix_n = int(self.adversarial_n * self.focus_strength)
        rand_n = self.adversarial_n - mix_n
        extra = [random.uniform(lo, hi) for _ in range(rand_n)]
        return top[:mix_n] + extra

    def update_focus(self, signal: float) -> None:
        self.focus_strength = clamp(self.focus_strength + 0.08 * signal, 0.05, 0.95)

    def make_train_val(self, top_fns: List[Callable[[float], float]]) -> Tuple[List[float], List[float]]:
        base = self.sample_base()
        adv = self.sample_adversarial(top_fns) if top_fns else []
        bnd = self.sample_boundary()
        reg = self.sample_regression()
        train = sorted(set(base + bnd))
        val = sorted(set(base + adv + bnd + reg))
        return train, val

    def sample_holdout(self, n: int = 128) -> List[float]:
        lo, hi = self.domain
        return [random.uniform(lo, hi) for _ in range(n)]

    def sample_stress(self, n: int = 160) -> List[float]:
        lo, hi = self.domain
        lo2, hi2 = lo * 1.35, hi * 1.35
        xs = [random.uniform(lo2, hi2) for _ in range(n)]
        xs += self.sample_boundary()
        return xs


# ---- bandit policy ----

@dataclass
class OperatorStats:
    name: str
    tries: int = 0
    wins: int = 0
    ema_gain: float = 0.0
    ema_decay: float = 0.93
    fail_streak: int = 0

    def ucb_score(self, total_tries: int, c: float = 1.0) -> float:
        t = max(1, self.tries)
        win_rate = (self.wins + 1) / (self.tries + 2)
        bonus = c * math.sqrt(math.log(total_tries + 1) / t)
        penalty = 0.05 * min(10, self.fail_streak)
        g = clamp(self.ema_gain if is_finite(self.ema_gain) else 0.0, -10.0, 10.0)
        score = (0.6 * win_rate + 0.4 * g) + bonus - penalty
        return score if is_finite(score) else 0.0

    def update(self, improved: bool, gain: float) -> None:
        self.tries += 1
        if improved:
            self.wins += 1
            self.fail_streak = 0
        else:
            self.fail_streak += 1
        g = clamp(gain if is_finite(gain) else 0.0, -10.0, 10.0)
        self.ema_gain = clamp(self.ema_gain * self.ema_decay + (1 - self.ema_decay) * g, -10.0, 10.0)

@dataclass
class OperatorBandit:
    ops: List[str]
    c: float = 1.0
    stats: Dict[str, OperatorStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for op in self.ops:
            if op not in self.stats:
                self.stats[op] = OperatorStats(op)

    def choose(self) -> str:
        total = sum(s.tries for s in self.stats.values()) + 1
        scored = [(self.stats[op].ucb_score(total, self.c), op) for op in self.ops]
        scored.sort(reverse=True)
        return scored[0][1]

    def update(self, op: str, improved: bool, gain: float) -> None:
        self.stats[op].update(improved, gain)


@dataclass
class OperatorSynthesizer:
    """
    Self-inventing operator generator.
    Proposes a new operator as a short program (sequence of primitive mutators),
    estimates expected gain, and if accepted, registers it as a new selectable operator.
    """
    min_len: int = 2
    max_len: int = 5
    trials: int = 6
    min_avg_gain: float = 1e-4
    max_attempts: int = 64
    allow_depth_delta: bool = True

    def _make_name(self, steps: List[str], depth_delta: int) -> str:
        payload = {"steps": steps, "max_depth_delta": depth_delta}
        h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
        return f"gen_op_{h}"

    def propose(self, existing: Dict[str, Any]) -> Dict[str, Any]:
        for _ in range(self.max_attempts):
            L = random.randint(self.min_len, self.max_len)
            steps = [random.choice(MUTATORS) for _ in range(L)]
            depth_delta = 0
            if self.allow_depth_delta and random.random() < 0.35:
                depth_delta = random.choice([-2, -1, 0, 1, 2])
            name = self._make_name(steps, depth_delta)
            if name not in existing and name not in MUTATORS:
                return {"name": name, "steps": steps, "max_depth_delta": depth_delta}
        return {"name": self._make_name([random.choice(MUTATORS), random.choice(MUTATORS)], 0),
                "steps": [random.choice(MUTATORS), random.choice(MUTATORS)],
                "max_depth_delta": 0}

    def evaluate_avg_gain(
        self,
        spec: Dict[str, Any],
        best_node: Node,
        best_obj: float,
        cfg_max_depth: int,
        target: Callable[[float], float],
        train_xs: List[float],
        val_xs: List[float],
        holdout_xs: List[float],
        stress_xs: List[float],
        contracts: List[Contract],
        probes: List[float],
        archive_sigs: List[Tuple[float, ...]],
        w_size: float,
        w_risk: float,
        w_novelty: float,
        gate_holdout_ref: float,
        gate_stress_ref: float,
        holdout_slack: float,
        stress_slack: float,
    ) -> float:
        lib = {spec["name"]: {"steps": spec["steps"], "max_depth_delta": spec.get("max_depth_delta", 0)}}
        gains: List[float] = []
        for _ in range(self.trials):
            cand = apply_operator(spec["name"], best_node, max_depth=cfg_max_depth, op_lib=lib)
            cand_score, cand_meta = evaluate_tree(
                cand, target, train_xs, val_xs, holdout_xs, stress_xs,
                contracts, probes, archive_sigs, w_size, w_risk, w_novelty
            )
            improved = (cand_score.objective < best_obj)
            if improved:
                if cand_meta["holdout_mse"] > gate_holdout_ref * holdout_slack:
                    improved = False
                if cand_meta["stress_mse"] > gate_stress_ref * stress_slack:
                    improved = False
            gain = (best_obj - cand_score.objective) if improved else 0.0
            gains.append(float(gain) if math.isfinite(float(gain)) else 0.0)
        return float(sum(gains) / max(1, len(gains)))

    def accept(self, avg_gain: float) -> bool:
        return avg_gain >= self.min_avg_gain

# ---- novelty ----


def behavior_signature(fn: Callable[[float], float], probes: List[float]) -> Tuple[float, ...]:
    out: List[float] = []
    for x in probes:
        try:
            y = fn(x)
        except Exception:
            y = float("nan")
        if not is_finite(y):
            y = 0.0
        out.append(clamp(float(y), -1e6, 1e6))
    return tuple(out)

def signature_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


# ---- mutations ----

def mutate_constants(root: Node, scale: float = 0.35) -> Node:
    r = root.clone()
    consts = [n for n in r.walk() if isinstance(n, Const)]
    if not consts:
        return r
    c = random.choice(consts)
    c.c = float(c.c) + random.gauss(0, scale)
    return simplify(r)

def replace_random_subtree(root: Node, max_depth: int) -> Node:
    r = root.clone()
    nodes = r.walk()
    target = random.choice(nodes)
    new_sub = random_tree(max_depth=max(1, max_depth - 1))
    if target is r:
        return simplify(new_sub)
    r.replace_child(target, new_sub)
    return simplify(r)

def insert_unary(root: Node) -> Node:
    r = root.clone()
    nodes = r.walk()
    target = random.choice(nodes)
    wrapped = Unary(random.choice(UNARY_OPS), target.clone())
    if target is r:
        return simplify(wrapped)
    r.replace_child(target, wrapped)
    return simplify(r)

def insert_binary(root: Node, max_depth: int) -> Node:
    r = root.clone()
    nodes = r.walk()
    target = random.choice(nodes)
    other = random_tree(max_depth=max(1, max_depth - 2))
    op = random.choice(BINARY_OPS)
    comb = Binary(op, target.clone(), other) if random.random() < 0.5 else Binary(op, other, target.clone())
    if target is r:
        return simplify(comb)
    r.replace_child(target, comb)
    return simplify(r)

def swap_two_subtrees(root: Node) -> Node:
    r = root.clone()
    nodes = [n for n in r.walk() if not isinstance(n, Var)]
    if len(nodes) < 2:
        return r
    a, b = random.sample(nodes, 2)
    if a is r or b is r:
        return r
    a_clone = a.clone()
    b_clone = b.clone()
    r.replace_child(a, b_clone)
    r.replace_child(b, a_clone)
    return simplify(r)

MUTATORS = ["mut_const", "replace_subtree", "insert_unary", "insert_binary", "swap_subtrees", "simplify"]

def apply_mutator(name: str, root: Node, max_depth: int) -> Node:
    if name == "mut_const":
        return mutate_constants(root)
    if name == "replace_subtree":
        return replace_random_subtree(root, max_depth=max_depth)
    if name == "insert_unary":
        return insert_unary(root)
    if name == "insert_binary":
        return insert_binary(root, max_depth=max_depth)
    if name == "swap_subtrees":
        return swap_two_subtrees(root)
    if name == "simplify":
        return simplify(root.clone())
    raise ValueError(name)


def apply_operator(op_name: str, root: Node, max_depth: int, op_lib: Dict[str, Any]) -> Node:
    """
    Apply either a primitive mutator (built-in) or a synthesized operator program.
    A synthesized operator is a short sequence of primitive mutators (a 'mutation program').
    """
    if op_name in MUTATORS:
        return apply_mutator(op_name, root, max_depth=max_depth)

    spec = op_lib.get(op_name)
    if not spec:
        return apply_mutator(random.choice(MUTATORS), root, max_depth=max_depth)

    steps = list(spec.get("steps", []))
    depth_delta = int(spec.get("max_depth_delta", 0))
    depth2 = max(2, int(max_depth) + depth_delta)

    r = root
    for st in steps:
        if st in MUTATORS:
            r = apply_mutator(st, r, max_depth=depth2)
        else:
            # Disallow nested synthesized operators to avoid recursion bombs.
            r = apply_mutator(random.choice(MUTATORS), r, max_depth=depth2)
    return r


def _rewrite_operators_block(src_text: str, new_dict: Dict[str, Any]) -> str:
    pattern = r"(# @@OPERATORS_LIB_START@@\s*OPERATORS_LIB\s*=\s*)({.*?})(\s*# @@OPERATORS_LIB_END@@)"
    mm = re.search(pattern, src_text, flags=re.S)
    if not mm:
        raise RuntimeError("Cannot find OPERATORS_LIB block markers.")
    prefix, _, suffix = mm.group(1), mm.group(2), mm.group(3)
    pretty = json.dumps(new_dict, ensure_ascii=False, indent=4, sort_keys=True)
    return src_text[:mm.start()] + prefix + pretty + suffix + src_text[mm.end():]


def load_ops_lib(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_ops_lib(path: Path, lib: Dict[str, Any]) -> None:
    path.write_text(json.dumps(lib, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


# ---- evaluation + archive ----

@dataclass
class Score:
    mse: float
    size: int
    risk: float
    novelty: float
    objective: float

def mse_on(fn: Callable[[float], float], target: Callable[[float], float], xs: List[float]) -> float:
    err = 0.0
    n = 0
    for x in xs:
        try:
            y = fn(x)
            t = target(x)
            if not (is_finite(y) and is_finite(t)):
                return float("inf")
            d = float(y) - float(t)
            err += d * d
            n += 1
        except Exception:
            return float("inf")
    return err / max(1, n)

def build_callable(root: Node) -> Callable[[float], float]:
    def _fn(x: float) -> float:
        return float(root.eval(x))
    return _fn

def evaluate_tree(
    root: Node,
    target: Callable[[float], float],
    train_xs: List[float],
    val_xs: List[float],
    holdout_xs: List[float],
    stress_xs: List[float],
    contracts: List[Contract],
    probes: List[float],
    archive_sigs: List[Tuple[float, ...]],
    w_size: float,
    w_risk: float,
    w_novelty: float,
) -> Tuple[Score, Dict[str, Any]]:
    fn = build_callable(root)

    # contract risk: count failures
    risk = 0.0
    for c in contracts:
        if not c.check(fn, val_xs):
            risk += 1.0

    train = mse_on(fn, target, train_xs)
    val = mse_on(fn, target, val_xs)
    hold = mse_on(fn, target, holdout_xs)
    stress = mse_on(fn, target, stress_xs)

    sig = behavior_signature(fn, probes)
    novelty = float(min(signature_distance(sig, s) for s in archive_sigs)) if archive_sigs else 1.0

    size = root.size()
    objective = val + w_size * size + w_risk * risk - w_novelty * novelty

    meta = {
        "train_mse": train,
        "val_mse": val,
        "holdout_mse": hold,
        "stress_mse": stress,
        "size": size,
        "risk": risk,
        "novelty": novelty,
        "signature": list(sig),
    }
    return Score(mse=val, size=size, risk=risk, novelty=novelty, objective=objective), meta

@dataclass
class Archive:
    outdir: Path
    def __post_init__(self) -> None:
        ensure_dir(self.outdir)
        ensure_dir(self.outdir / "snapshots")
        self.log_path = self.outdir / "archive.jsonl"
        self.best_path = self.outdir / "best.json"
    def append(self, rec: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    def save_best(self, rec: Dict[str, Any]) -> None:
        self.best_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    def snapshot(self, rec: Dict[str, Any]) -> None:
        sid = rec.get("id", "unknown")
        (self.outdir / "snapshots" / f"{sid}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )

def node_to_dict(n: Node) -> Dict[str, Any]:
    if isinstance(n, Var):
        return {"t": "Var"}
    if isinstance(n, Const):
        return {"t": "Const", "c": n.c}
    if isinstance(n, Unary):
        return {"t": "Unary", "op": n.op, "a": node_to_dict(n.a)}
    if isinstance(n, Binary):
        return {"t": "Binary", "op": n.op, "a": node_to_dict(n.a), "b": node_to_dict(n.b)}
    raise TypeError(type(n))

def dict_to_node(d: Dict[str, Any]) -> Node:
    t = d["t"]
    if t == "Var":
        return Var()
    if t == "Const":
        return Const(float(d["c"]))
    if t == "Unary":
        return Unary(d["op"], dict_to_node(d["a"]))
    if t == "Binary":
        return Binary(d["op"], dict_to_node(d["a"]), dict_to_node(d["b"]))
    raise ValueError(t)


# ---- tasks ----

def get_target(task: str) -> Callable[[float], float]:
    if task == "sin":
        return lambda x: math.sin(x) + 0.25 * x
    if task == "poly":
        return lambda x: 0.3 * x * x - 0.2 * x + 0.7
    if task == "mix":
        return lambda x: math.cos(1.5 * x) + 0.1 * (x ** 3) - 0.4 * x
    if task == "abs":
        return lambda x: abs(x) + 0.05 * x
    raise ValueError(f"Unknown task: {task}")


# ---- SCIG config + run ----

@dataclass
class SCIGConfig:
    steps: int = 400
    max_depth: int = 6
    seed: int = 7
    task: str = "mix"
    outdir: str = "runs/scig"
    w_size: float = 1e-3
    w_risk: float = 0.25
    w_novelty: float = 1e-2
    accept_margin: float = 1e-9
    holdout_slack: float = 1.20
    stress_slack: float = 1.35
    enable_self_ops: bool = True
    ops_synth_every: int = 40
    ops_trials: int = 6
    ops_min_avg_gain: float = 1e-4
    ops_lib_path: str = ""
    self_patch_ops: bool = False
    domain_lo: float = -3.0
    domain_hi: float = 3.0

def scig_load_best(outdir: Path) -> Optional[Dict[str, Any]]:
    p = outdir / "best.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def scig_run(cfg: SCIGConfig) -> None:
    set_seed(cfg.seed)
    outdir = Path(cfg.outdir)
    ensure_dir(outdir)

    target = get_target(cfg.task)
    forge = TestForgePlus(domain=(cfg.domain_lo, cfg.domain_hi))

    probes = [random.uniform(cfg.domain_lo, cfg.domain_hi) for _ in range(24)]
    contracts = [
        contract_finite_and_bounded(1e6),
        contract_lipschitz_soft(1e4),
        contract_smooth_probe(2e5, 1e-4),
    ]
    archive = Archive(outdir=outdir)

    # ---- self-installed operators (learned mutation programs) ----
    ops_lib_path = Path(cfg.ops_lib_path) if cfg.ops_lib_path else (outdir / "operators.json")
    op_lib_disk = load_ops_lib(ops_lib_path)
    op_lib = dict(OPERATORS_LIB)
    op_lib.update(op_lib_disk)
    active_ops = list(MUTATORS)
    if cfg.enable_self_ops:
        for k in sorted(op_lib.keys()):
            if k not in active_ops and k not in MUTATORS:
                active_ops.append(k)

    bandit = OperatorBandit(ops=active_ops, c=1.0)

    best_rec = scig_load_best(outdir)
    if best_rec:
        best = dict_to_node(best_rec["tree"])
        best_meta = best_rec["meta"]
        best_score = Score(**best_rec["score"])
        best_id = best_rec["id"]
        archive_sigs = [tuple(best_meta.get("signature", []))] if best_meta.get("signature") else []
    else:
        best = random_tree(max_depth=cfg.max_depth)
        train_xs, val_xs = forge.make_train_val([])
        holdout_xs = forge.sample_holdout()
        stress_xs = forge.sample_stress()
        best_score, best_meta = evaluate_tree(
            best, target, train_xs, val_xs, holdout_xs, stress_xs,
            contracts, probes, [], cfg.w_size, cfg.w_risk, cfg.w_novelty,
        )
        best_id = sha256_text(best.to_str() + str(cfg.seed))[:12]
        archive_sigs = [tuple(best_meta["signature"])]
        archive.save_best({"id": best_id, "ts": now_ts(), "tree": node_to_dict(best), "score": asdict(best_score), "meta": best_meta})

    print(f"[SCIG] start best_id={best_id} obj={best_score.objective:.6g} val={best_meta['val_mse']:.6g}")
    synth = OperatorSynthesizer(trials=int(cfg.ops_trials), min_avg_gain=float(cfg.ops_min_avg_gain))


    best_node = best
    best_meta_local = best_meta
    best_score_local = best_score
    best_id_local = best_id

    top_fns: List[Callable[[float], float]] = [build_callable(best_node)]
    accepted = 0
    rejected = 0

    for step in range(1, cfg.steps + 1):
        train_xs, val_xs = forge.make_train_val(top_fns[:3])
        holdout_xs = forge.sample_holdout()
        stress_xs = forge.sample_stress()

        # Periodically invent a new operator program and install it (disk + optionally patch source)
        if cfg.enable_self_ops and cfg.ops_synth_every > 0 and (step % int(cfg.ops_synth_every) == 0):
            spec = synth.propose(op_lib)
            avg_gain = synth.evaluate_avg_gain(
                spec=spec,
                best_node=best_node,
                best_obj=float(best_score_local.objective),
                cfg_max_depth=int(cfg.max_depth),
                target=target,
                train_xs=train_xs,
                val_xs=val_xs,
                holdout_xs=holdout_xs,
                stress_xs=stress_xs,
                contracts=contracts,
                probes=probes,
                archive_sigs=archive_sigs,
                w_size=float(cfg.w_size),
                w_risk=float(cfg.w_risk),
                w_novelty=float(cfg.w_novelty),
                gate_holdout_ref=float(best_meta_local["holdout_mse"]),
                gate_stress_ref=float(best_meta_local["stress_mse"]),
                holdout_slack=float(cfg.holdout_slack),
                stress_slack=float(cfg.stress_slack),
            )
            if synth.accept(avg_gain):
                op_name = str(spec["name"])
                op_lib[op_name] = {"steps": list(spec["steps"]), "max_depth_delta": int(spec.get("max_depth_delta", 0))}
                save_ops_lib(ops_lib_path, op_lib)

                if op_name not in bandit.ops:
                    bandit.ops.append(op_name)
                    bandit.stats[op_name] = OperatorStats(op_name)

                archive.append({
                    "ts": now_ts(),
                    "step": step,
                    "op": "SELF_INSTALL",
                    "accepted": True,
                    "operator_name": op_name,
                    "operator_spec": op_lib[op_name],
                    "avg_gain": float(avg_gain),
                    "note": "Synthesized operator installed."
                })
                print(f"[SCIG][SELF] installed {op_name} avg_gain={avg_gain:.3g} steps={spec['steps']} depth_delta={spec.get('max_depth_delta',0)}")

                if cfg.self_patch_ops:
                    try:
                        src_text = Path(__file__).read_text(encoding="utf-8")
                        patched = _rewrite_operators_block(src_text, op_lib)
                        Path(__file__).write_text(patched, encoding="utf-8")
                        print("[SCIG][SELF] patched source OPERATORS_LIB block in-place.")
                    except Exception as e:
                        print(f"[SCIG][SELF] source patch failed: {e}")
            else:
                archive.append({
                    "ts": now_ts(),
                    "step": step,
                    "op": "SELF_INSTALL",
                    "accepted": False,
                    "operator_spec": {"steps": list(spec["steps"]), "max_depth_delta": int(spec.get("max_depth_delta", 0))},
                    "avg_gain": float(avg_gain),
                    "note": "Synthesized operator rejected."
                })

        op = bandit.choose()
        cand = apply_operator(op, best_node, max_depth=cfg.max_depth, op_lib=op_lib)
        cand_score, cand_meta = evaluate_tree(
            cand, target, train_xs, val_xs, holdout_xs, stress_xs,
            contracts, probes, archive_sigs, cfg.w_size, cfg.w_risk, cfg.w_novelty,
        )

        improved = (cand_score.objective + cfg.accept_margin) < best_score_local.objective

        # gating: avoid catastrophic overfit
        if improved:
            if cand_meta["holdout_mse"] > best_meta_local["holdout_mse"] * cfg.holdout_slack:
                improved = False
            if cand_meta["stress_mse"] > best_meta_local["stress_mse"] * cfg.stress_slack:
                improved = False

        gain = best_score_local.objective - cand_score.objective
        bandit.update(op, improved, gain)

        archive.append({
            "ts": now_ts(),
            "step": step,
            "op": op,
            "parent": best_id_local,
            "candidate_tree": node_to_dict(cand),
            "candidate_score": asdict(cand_score),
            "candidate_meta": cand_meta,
            "accepted": bool(improved),
        })

        if improved:
            accepted += 1
            best_node = cand
            best_score_local = cand_score
            best_meta_local = cand_meta
            best_id_local = sha256_text(cand.to_str() + f"{cfg.seed}:{step}")[:12]
            archive_sigs.append(tuple(cand_meta["signature"]))
            top_fns = [build_callable(best_node)] + top_fns[:4]
            forge.add_regression_points([x for x in val_xs if random.random() < 0.07])
            forge.update_focus(+0.2 if cand_meta["risk"] > 0 else -0.05)

            best_blob = {"id": best_id_local, "ts": now_ts(), "tree": node_to_dict(best_node), "score": asdict(best_score_local), "meta": best_meta_local}
            archive.save_best(best_blob)
            archive.snapshot(best_blob)
        else:
            rejected += 1

        if step % 25 == 0 or step == cfg.steps:
            s = bandit.stats
            tot = sum(st.tries for st in s.values()) + 1
            top_ops = sorted([(st.ucb_score(tot), k, st.tries, st.wins) for k, st in s.items()], reverse=True)[:3]
            print(f"[SCIG] step={step} acc={accepted} rej={rejected} best_obj={best_score_local.objective:.6g} "
                  f"val={best_meta_local['val_mse']:.6g} hold={best_meta_local['holdout_mse']:.6g} stress={best_meta_local['stress_mse']:.6g} "
                  f"top_ops={[ (k,t,w) for _,k,t,w in top_ops ]}")

    print(f"[SCIG] done best_id={best_id_local} obj={best_score_local.objective:.6g} val={best_meta_local['val_mse']:.6g}")
    print(f"[SCIG] outdir {outdir.resolve()}")


# =============================================================================
# Engine 2: Ouroboros (meta-loop to tune SCIG hyperparameters)
# =============================================================================

# @@OUROBOROS_TUNED_START@@
OUROBOROS_TUNED = {
    "holdout_slack": 1.2,
    "max_depth": 6,
    "stress_slack": 1.35,
    "w_novelty": 0.01,
    "w_risk": 0.25,
    "w_size": 0.001
}
# @@OUROBOROS_TUNED_END@@

# @@OPERATORS_LIB_START@@
OPERATORS_LIB = {
    "gen_op_012e3f2583": {
        "max_depth_delta": -1,
        "steps": [
            "insert_unary",
            "replace_subtree",
            "insert_unary",
            "replace_subtree"
        ]
    },
    "gen_op_065db36941": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "insert_unary",
            "mut_const",
            "mut_const"
        ]
    },
    "gen_op_09ac2acbd6": {
        "max_depth_delta": 0,
        "steps": [
            "swap_subtrees",
            "replace_subtree"
        ]
    },
    "gen_op_0af8eeea54": {
        "max_depth_delta": 0,
        "steps": [
            "mut_const",
            "replace_subtree"
        ]
    },
    "gen_op_0d6548ef33": {
        "max_depth_delta": 2,
        "steps": [
            "insert_binary",
            "simplify",
            "insert_unary",
            "simplify"
        ]
    },
    "gen_op_10cfc609b4": {
        "max_depth_delta": 0,
        "steps": [
            "swap_subtrees",
            "simplify",
            "insert_binary",
            "simplify"
        ]
    },
    "gen_op_1d08c7adaf": {
        "max_depth_delta": 0,
        "steps": [
            "simplify",
            "simplify",
            "insert_binary",
            "replace_subtree"
        ]
    },
    "gen_op_1e419bdd35": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "swap_subtrees"
        ]
    },
    "gen_op_1fb82854a5": {
        "max_depth_delta": 0,
        "steps": [
            "mut_const",
            "insert_binary",
            "simplify",
            "simplify",
            "swap_subtrees"
        ]
    },
    "gen_op_21dc1aa040": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "mut_const"
        ]
    },
    "gen_op_240b47edf7": {
        "max_depth_delta": 0,
        "steps": [
            "simplify",
            "replace_subtree"
        ]
    },
    "gen_op_282d07a153": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "insert_unary",
            "simplify"
        ]
    },
    "gen_op_2859d9f6e1": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "mut_const",
            "simplify",
            "replace_subtree"
        ]
    },
    "gen_op_2c4c0c0926": {
        "max_depth_delta": 0,
        "steps": [
            "swap_subtrees",
            "simplify",
            "replace_subtree"
        ]
    },
    "gen_op_2c707910fb": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "insert_binary",
            "insert_binary"
        ]
    },
    "gen_op_2dd818af87": {
        "max_depth_delta": 0,
        "steps": [
            "swap_subtrees",
            "insert_binary",
            "simplify",
            "swap_subtrees",
            "insert_binary"
        ]
    },
    "gen_op_324853dc1a": {
        "max_depth_delta": 0,
        "steps": [
            "mut_const",
            "insert_binary",
            "mut_const",
            "simplify",
            "mut_const"
        ]
    },
    "gen_op_356c1bda10": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "insert_unary",
            "insert_binary"
        ]
    },
    "gen_op_38a9602e97": {
        "max_depth_delta": 0,
        "steps": [
            "simplify",
            "insert_unary",
            "insert_binary",
            "swap_subtrees",
            "replace_subtree"
        ]
    },
    "gen_op_4003a59d00": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "simplify",
            "mut_const",
            "insert_binary"
        ]
    },
    "gen_op_40c0ac7ba8": {
        "max_depth_delta": -1,
        "steps": [
            "simplify",
            "swap_subtrees",
            "replace_subtree",
            "mut_const",
            "replace_subtree"
        ]
    },
    "gen_op_4179c12b32": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "simplify",
            "mut_const"
        ]
    },
    "gen_op_43c8509037": {
        "max_depth_delta": 0,
        "steps": [
            "mut_const",
            "insert_binary",
            "swap_subtrees",
            "insert_binary",
            "insert_unary"
        ]
    },
    "gen_op_4425f3a67e": {
        "max_depth_delta": 2,
        "steps": [
            "simplify",
            "mut_const",
            "insert_binary",
            "simplify"
        ]
    },
    "gen_op_483b75cee4": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "simplify",
            "insert_unary",
            "swap_subtrees"
        ]
    },
    "gen_op_4bd33da888": {
        "max_depth_delta": 2,
        "steps": [
            "replace_subtree",
            "insert_binary",
            "replace_subtree",
            "mut_const",
            "mut_const"
        ]
    },
    "gen_op_4e80d2e325": {
        "max_depth_delta": 0,
        "steps": [
            "swap_subtrees",
            "insert_binary",
            "insert_unary"
        ]
    },
    "gen_op_4f7d695069": {
        "max_depth_delta": -1,
        "steps": [
            "insert_binary",
            "mut_const"
        ]
    },
    "gen_op_57b87eccb3": {
        "max_depth_delta": 0,
        "steps": [
            "simplify",
            "simplify",
            "simplify",
            "mut_const",
            "insert_binary"
        ]
    },
    "gen_op_58cad40d8f": {
        "max_depth_delta": 1,
        "steps": [
            "replace_subtree",
            "insert_binary",
            "mut_const"
        ]
    },
    "gen_op_5b017be932": {
        "max_depth_delta": 0,
        "steps": [
            "insert_unary",
            "replace_subtree",
            "simplify",
            "mut_const"
        ]
    },
    "gen_op_66d7d9c2c0": {
        "max_depth_delta": 1,
        "steps": [
            "replace_subtree",
            "mut_const"
        ]
    },
    "gen_op_685c62c796": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "replace_subtree"
        ]
    },
    "gen_op_6c2e9b45e0": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "swap_subtrees"
        ]
    },
    "gen_op_6c53f4897e": {
        "max_depth_delta": 0,
        "steps": [
            "swap_subtrees",
            "replace_subtree",
            "replace_subtree",
            "replace_subtree",
            "insert_binary"
        ]
    },
    "gen_op_6ea9efc21e": {
        "max_depth_delta": -2,
        "steps": [
            "mut_const",
            "mut_const",
            "replace_subtree",
            "insert_unary"
        ]
    },
    "gen_op_6f9c0c7ef1": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "insert_unary",
            "mut_const",
            "replace_subtree"
        ]
    },
    "gen_op_762d0c06c5": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "mut_const",
            "mut_const"
        ]
    },
    "gen_op_79b51ea94f": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "mut_const",
            "replace_subtree",
            "insert_binary"
        ]
    },
    "gen_op_7aa9982c93": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "replace_subtree"
        ]
    },
    "gen_op_7d3a8e56ff": {
        "max_depth_delta": 1,
        "steps": [
            "replace_subtree",
            "mut_const",
            "insert_binary"
        ]
    },
    "gen_op_84afe698c8": {
        "max_depth_delta": 0,
        "steps": [
            "mut_const",
            "insert_binary",
            "insert_unary",
            "insert_unary"
        ]
    },
    "gen_op_86eb5d0493": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "insert_unary",
            "replace_subtree",
            "mut_const",
            "insert_unary"
        ]
    },
    "gen_op_8f0a19ad8d": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "replace_subtree",
            "replace_subtree"
        ]
    },
    "gen_op_9326e10b9a": {
        "max_depth_delta": 2,
        "steps": [
            "insert_unary",
            "insert_unary",
            "insert_binary"
        ]
    },
    "gen_op_93e5b69022": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "insert_binary",
            "insert_binary"
        ]
    },
    "gen_op_98185c0462": {
        "max_depth_delta": 0,
        "steps": [
            "insert_unary",
            "insert_unary",
            "insert_binary",
            "swap_subtrees",
            "simplify"
        ]
    },
    "gen_op_9c327602e3": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "replace_subtree",
            "insert_binary",
            "mut_const"
        ]
    },
    "gen_op_9e522fd8cf": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "mut_const",
            "mut_const"
        ]
    },
    "gen_op_a4430723f2": {
        "max_depth_delta": 0,
        "steps": [
            "simplify",
            "replace_subtree",
            "insert_unary",
            "insert_unary"
        ]
    },
    "gen_op_a93a49b726": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "mut_const",
            "insert_unary"
        ]
    },
    "gen_op_b587f6146b": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "mut_const"
        ]
    },
    "gen_op_c0d7f9f4f4": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "replace_subtree",
            "simplify",
            "mut_const"
        ]
    },
    "gen_op_c6eef560b8": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "insert_binary"
        ]
    },
    "gen_op_ce189b9e5a": {
        "max_depth_delta": 0,
        "steps": [
            "insert_unary",
            "replace_subtree",
            "insert_binary",
            "simplify"
        ]
    },
    "gen_op_ce5abb631a": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "swap_subtrees",
            "simplify"
        ]
    },
    "gen_op_d4652e75ec": {
        "max_depth_delta": 0,
        "steps": [
            "simplify",
            "insert_binary",
            "mut_const",
            "insert_unary"
        ]
    },
    "gen_op_db5a5c8652": {
        "max_depth_delta": 0,
        "steps": [
            "insert_binary",
            "swap_subtrees",
            "replace_subtree"
        ]
    },
    "gen_op_e07d04ac85": {
        "max_depth_delta": 2,
        "steps": [
            "mut_const",
            "insert_unary",
            "mut_const",
            "replace_subtree",
            "mut_const"
        ]
    },
    "gen_op_e674814e70": {
        "max_depth_delta": 0,
        "steps": [
            "mut_const",
            "insert_unary",
            "insert_binary",
            "swap_subtrees",
            "simplify"
        ]
    },
    "gen_op_ea7a181a1c": {
        "max_depth_delta": 0,
        "steps": [
            "simplify",
            "insert_unary",
            "insert_binary"
        ]
    },
    "gen_op_ef1eb59447": {
        "max_depth_delta": 0,
        "steps": [
            "simplify",
            "swap_subtrees",
            "insert_binary",
            "insert_binary",
            "mut_const"
        ]
    },
    "gen_op_f3874f0df0": {
        "max_depth_delta": 0,
        "steps": [
            "replace_subtree",
            "replace_subtree",
            "mut_const",
            "insert_unary"
        ]
    },
    "gen_op_fabc7181c2": {
        "max_depth_delta": 0,
        "steps": [
            "mut_const",
            "mut_const",
            "insert_binary",
            "insert_unary",
            "swap_subtrees"
        ]
    }
}
# @@OPERATORS_LIB_END@@


def _rewrite_ouroboros_block(src_text: str, new_dict: Dict[str, Any]) -> str:
    pattern = r"(# @@OUROBOROS_TUNED_START@@\s*OUROBOROS_TUNED\s*=\s*)({.*?})(\s*# @@OUROBOROS_TUNED_END@@)"
    m = re.search(pattern, src_text, flags=re.S)
    if not m:
        raise RuntimeError("Cannot find OUROBOROS_TUNED block markers.")
    prefix, _, suffix = m.group(1), m.group(2), m.group(3)
    pretty = json.dumps(new_dict, ensure_ascii=False, indent=4, sort_keys=True)
    return src_text[:m.start()] + prefix + pretty + suffix + src_text[m.end():]

def ouroboros_run(args: argparse.Namespace) -> None:
    base = dict(OUROBOROS_TUNED)
    set_seed(args.seed)

    def propose(d: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(d)
        for k, lo, hi in [("w_size", 1e-5, 5e-2), ("w_risk", 0.05, 1.0), ("w_novelty", 1e-6, 5e-2)]:
            if random.random() < 0.8:
                x = math.log10(max(lo, float(p.get(k, lo)), 1e-12))
                x += random.gauss(0, 0.35)
                p[k] = float(10 ** clamp(x, math.log10(lo), math.log10(hi)))
        if random.random() < 0.7:
            p["holdout_slack"] = float(clamp(p["holdout_slack"] + random.gauss(0, 0.05), 1.02, 1.60))
        if random.random() < 0.7:
            p["stress_slack"] = float(clamp(p["stress_slack"] + random.gauss(0, 0.07), 1.05, 2.0))
        if random.random() < 0.6:
            p["max_depth"] = int(clamp(int(p["max_depth"] + random.choice([-1, 1])), 4, 10))
        return p

    def probe(h: Dict[str, Any], trial_id: int) -> float:
        outdir = Path(args.outdir) / f"ouro_probe_{trial_id:03d}"
        if outdir.exists():
            shutil.rmtree(outdir, ignore_errors=True)
        cfg = SCIGConfig(
            steps=args.probe_steps,
            seed=args.seed + trial_id * 13,
            task=args.task,
            outdir=str(outdir),
            max_depth=int(h["max_depth"]),
            w_size=float(h["w_size"]),
            w_risk=float(h["w_risk"]),
            w_novelty=float(h["w_novelty"]),
            holdout_slack=float(h["holdout_slack"]),
            stress_slack=float(h["stress_slack"]),
enable_self_ops=enable_self_ops,
ops_synth_every=int(getattr(args, "ops_synth_every", 40)),
ops_trials=int(getattr(args, "ops_trials", 6)),
ops_min_avg_gain=float(getattr(args, "ops_min_avg_gain", 1e-4)),
ops_lib_path=str(getattr(args, "ops_lib_path", "")),
self_patch_ops=bool(getattr(args, "self_patch_ops", False)),
domain_lo=args.domain_lo,

            domain_hi=args.domain_hi,
        )
        scig_run(cfg)
        best = scig_load_best(outdir)
        return float(best["score"]["objective"]) if best else float("inf")

    best_h = dict(base)
    best_obj = probe(best_h, 0)
    print(f"[OURO] baseline obj={best_obj:.6g} tuned={best_h}")

    for t in range(1, args.trials + 1):
        cand = propose(best_h if random.random() < 0.6 else base)
        obj = probe(cand, t)
        if obj < best_obj:
            print(f"[OURO] improve t={t} obj={best_obj:.6g} -> {obj:.6g} cand={cand}")
            best_obj = obj
            best_h = cand
        else:
            print(f"[OURO] t={t} obj={obj:.6g} (no)")

    print(f"[OURO] best obj={best_obj:.6g} tuned={best_h}")

    if args.emit_patched:
        src = Path(__file__).read_text(encoding="utf-8")
        patched = _rewrite_ouroboros_block(src, best_h)
        out_file = Path(args.emit_patched)
        out_file.write_text(patched, encoding="utf-8")
        print(f"[OURO] wrote patched file: {out_file.resolve()}")


# =============================================================================
# Engine 3: L7 (NumPy optional)
# =============================================================================

def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for --engine l7, but import failed.")

def safe_add(a, b): return np.nan_to_num(np.add(a, b), nan=0.0, posinf=1e10, neginf=-1e10)
def safe_sub(a, b): return np.nan_to_num(np.subtract(a, b), nan=0.0, posinf=1e10, neginf=-1e10)
def safe_mul(a, b):
    a = np.clip(a, -1e5, 1e5); b = np.clip(b, -1e5, 1e5)
    return np.nan_to_num(np.multiply(a, b), nan=0.0, posinf=1e10, neginf=-1e10)
def safe_div(a, b): return np.nan_to_num(np.divide(a, b + 1e-10), nan=0.0, posinf=1e10, neginf=-1e10)
def safe_pow(a, b):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        b2 = np.clip(b, -3, 3)
        base = np.where(a < 0, np.abs(a), a) + 1e-10
        return np.nan_to_num(np.power(base, b2), nan=0.0, posinf=1e10, neginf=-1e10)
def safe_sin(a): return np.nan_to_num(np.sin(a), nan=0.0)
def safe_cos(a): return np.nan_to_num(np.cos(a), nan=0.0)
def safe_exp(a): return np.exp(np.clip(a, -10, 10))
def safe_log(a): return np.log(np.abs(a) + 1e-10)
def safe_abs(a): return np.abs(a)
def safe_neg(a): return np.negative(a)
def safe_sq(a): return np.square(np.clip(a, -1e5, 1e5))

L7_UNARY = ["sin", "cos", "exp", "log", "abs", "neg", "sq"]
L7_BINARY = ["add", "sub", "mul", "div", "pow"]

def l7_apply_unary(op: str, a):
    if op == "sin": return safe_sin(a)
    if op == "cos": return safe_cos(a)
    if op == "exp": return safe_exp(a)
    if op == "log": return safe_log(a)
    if op == "abs": return safe_abs(a)
    if op == "neg": return safe_neg(a)
    if op == "sq":  return safe_sq(a)
    raise ValueError(op)

def l7_apply_binary(op: str, a, b):
    if op == "add": return safe_add(a, b)
    if op == "sub": return safe_sub(a, b)
    if op == "mul": return safe_mul(a, b)
    if op == "div": return safe_div(a, b)
    if op == "pow": return safe_pow(a, b)
    raise ValueError(op)

@dataclass
class L7Expr:
    t: str
    v: Any = None
    a: Optional["L7Expr"] = None
    b: Optional["L7Expr"] = None

    def count_ops(self) -> int:
        if self.t in ("var", "const"):
            return 0
        if self.t == "un":
            return 1 + (self.a.count_ops() if self.a else 0)
        if self.t == "bin":
            return 1 + (self.a.count_ops() if self.a else 0) + (self.b.count_ops() if self.b else 0)
        return 0

    def clone(self) -> "L7Expr":
        return L7Expr(self.t, self.v, self.a.clone() if self.a else None, self.b.clone() if self.b else None)

    def eval(self, x):
        if self.t == "var":
            return x
        if self.t == "const":
            return np.full_like(x, float(self.v))
        if self.t == "un":
            return l7_apply_unary(str(self.v), self.a.eval(x))
        if self.t == "bin":
            return l7_apply_binary(str(self.v), self.a.eval(x), self.b.eval(x))
        return np.zeros_like(x)

    def to_code(self) -> str:
        if self.t == "var":
            return "x"
        if self.t == "const":
            return str(float(self.v))
        if self.t == "un":
            m = {"sin":"safe_sin","cos":"safe_cos","exp":"safe_exp","log":"safe_log","abs":"safe_abs","neg":"safe_neg","sq":"safe_sq"}[str(self.v)]
            return f"{m}({self.a.to_code()})"
        if self.t == "bin":
            m = {"add":"safe_add","sub":"safe_sub","mul":"safe_mul","div":"safe_div","pow":"safe_pow"}[str(self.v)]
            return f"{m}({self.a.to_code()}, {self.b.to_code()})"
        return "0.0"

def l7_rand_leaf() -> L7Expr:
    return L7Expr("var") if random.random() < 0.6 else L7Expr("const", v=round(random.uniform(-2.0, 2.0), 4))

def l7_rand_expr(max_depth: int) -> L7Expr:
    if max_depth <= 1:
        return l7_rand_leaf()
    r = random.random()
    if r < 0.30:
        return l7_rand_leaf()
    if r < 0.60:
        return L7Expr("un", v=random.choice(L7_UNARY), a=l7_rand_expr(max_depth - 1))
    return L7Expr("bin", v=random.choice(L7_BINARY), a=l7_rand_expr(max_depth - 1), b=l7_rand_expr(max_depth - 1))

def l7_pick(root: L7Expr) -> List[L7Expr]:
    out = [root]
    if root.a: out += l7_pick(root.a)
    if root.b: out += l7_pick(root.b)
    return out

def l7_mutate(root: L7Expr, max_depth: int) -> L7Expr:
    r = root.clone()
    tgt = random.choice(l7_pick(r))
    choice = random.random()
    if choice < 0.35:
        if tgt.t == "const":
            tgt.v = round(float(tgt.v) + random.gauss(0, 0.4), 4)
        else:
            tgt.t = "const"; tgt.v = round(random.uniform(-3.0, 3.0), 4); tgt.a = None; tgt.b = None
    elif choice < 0.65:
        new = L7Expr("un", v=random.choice(L7_UNARY), a=tgt.clone())
        tgt.t, tgt.v, tgt.a, tgt.b = new.t, new.v, new.a, None
    else:
        new = l7_rand_expr(max_depth)
        tgt.t, tgt.v, tgt.a, tgt.b = new.t, new.v, new.a, new.b
    return r

def l7_mse(expr: L7Expr, x: "np.ndarray", y: "np.ndarray") -> float:
    yp = expr.eval(x)
    if not np.all(np.isfinite(yp)):
        return float("inf")
    d = yp - y
    return float(np.mean(d * d))

@dataclass
class L7State:
    version: int = 1
    seed: int = 7
    step: int = 0
    best_expr: Dict[str, Any] = field(default_factory=dict)
    best_mse: float = float("inf")
    best_ops: int = 10**9
    policy: Dict[str, float] = field(default_factory=lambda: {"mutate": 1.0, "reset": 1.0})
    wins: Dict[str, int] = field(default_factory=lambda: {"mutate": 0, "reset": 0})
    tries: Dict[str, int] = field(default_factory=lambda: {"mutate": 0, "reset": 0})

def l7_expr_to_dict(e: L7Expr) -> Dict[str, Any]:
    return {"t": e.t, "v": e.v, "a": l7_expr_to_dict(e.a) if e.a else None, "b": l7_expr_to_dict(e.b) if e.b else None}

def l7_dict_to_expr(d: Dict[str, Any]) -> L7Expr:
    return L7Expr(d["t"], d.get("v"), l7_dict_to_expr(d["a"]) if d.get("a") else None, l7_dict_to_expr(d["b"]) if d.get("b") else None)

def l7_load_state(outdir: Path) -> Optional[L7State]:
    p = outdir / "l7_state.json"
    if not p.exists():
        return None
    try:
        return L7State(**json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return None

def l7_save_state(outdir: Path, st: L7State) -> None:
    (outdir / "l7_state.json").write_text(json.dumps(asdict(st), ensure_ascii=False, indent=2), encoding="utf-8")

def emit_l7_candidate(expr: L7Expr, outdir: Path) -> None:
    out = outdir / "best_candidate.py"
    out.write_text(
        "# Auto-emitted candidate (L7)\n"
        "import numpy as np\n\n"
        "def safe_add(a,b): return np.nan_to_num(np.add(a,b), nan=0.0, posinf=1e10, neginf=-1e10)\n"
        "def safe_sub(a,b): return np.nan_to_num(np.subtract(a,b), nan=0.0, posinf=1e10, neginf=-1e10)\n"
        "def safe_mul(a,b):\n"
        "    a=np.clip(a,-1e5,1e5); b=np.clip(b,-1e5,1e5)\n"
        "    return np.nan_to_num(np.multiply(a,b), nan=0.0, posinf=1e10, neginf=-1e10)\n"
        "def safe_div(a,b): return np.nan_to_num(np.divide(a,b+1e-10), nan=0.0, posinf=1e10, neginf=-1e10)\n"
        "def safe_pow(a,b):\n"
        "    with np.errstate(invalid='ignore', divide='ignore', over='ignore', under='ignore'):\n"
        "        b2=np.clip(b,-3,3); base=np.where(a<0,np.abs(a),a)+1e-10\n"
        "        return np.nan_to_num(np.power(base,b2), nan=0.0, posinf=1e10, neginf=-1e10)\n"
        "def safe_sin(a): return np.nan_to_num(np.sin(a), nan=0.0)\n"
        "def safe_cos(a): return np.nan_to_num(np.cos(a), nan=0.0)\n"
        "def safe_exp(a): return np.exp(np.clip(a,-10,10))\n"
        "def safe_log(a): return np.log(np.abs(a)+1e-10)\n"
        "def safe_abs(a): return np.abs(a)\n"
        "def safe_neg(a): return np.negative(a)\n"
        "def safe_sq(a): return np.square(np.clip(a,-1e5,1e5))\n\n"
        "def candidate(x):\n"
        f"    return {expr.to_code()}\n",
        encoding="utf-8"
    )

def l7_run(args: argparse.Namespace) -> None:
    _require_numpy()
    set_seed(args.seed)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    tgt = get_target(args.task)
    x = np.linspace(args.domain_lo, args.domain_hi, args.n_points).astype(np.float64)
    y = np.array([tgt(float(v)) for v in x], dtype=np.float64)

    st = l7_load_state(outdir) if args.resume else None
    if st and st.best_expr:
        best = l7_dict_to_expr(st.best_expr)
        best_mse = float(st.best_mse)
        best_ops = int(st.best_ops)
        step0 = int(st.step)
    else:
        best = l7_rand_expr(args.max_depth)
        best_mse = l7_mse(best, x, y)
        best_ops = best.count_ops()
        st = L7State(seed=args.seed, step=0, best_expr=l7_expr_to_dict(best), best_mse=best_mse, best_ops=best_ops)
        step0 = 0

    def pick_action() -> str:
        w = st.policy
        s = w["mutate"] + w["reset"]
        r = random.random() * s
        return "mutate" if r < w["mutate"] else "reset"

    print(f"[L7] start step={step0} best_mse={best_mse:.6g} ops={best_ops} policy={st.policy}")
    for k in range(1, args.steps + 1):
        st.step = step0 + k
        action = pick_action()
        st.tries[action] += 1

        cand = l7_rand_expr(args.max_depth) if action == "reset" else l7_mutate(best, args.max_depth)
        cmse = l7_mse(cand, x, y)
        cops = cand.count_ops()

        improved = (cmse < best_mse) or (abs(cmse - best_mse) < 1e-12 and cops < best_ops)
        if improved:
            best = cand
            best_mse = cmse
            best_ops = cops
            st.wins[action] += 1
            st.policy[action] = min(8.0, st.policy[action] * 1.03 + 0.02)
            other = "reset" if action == "mutate" else "mutate"
            st.policy[other] = max(0.25, st.policy[other] * 0.995)
            st.best_expr = l7_expr_to_dict(best)
            st.best_mse = float(best_mse)
            st.best_ops = int(best_ops)

        if k % args.save_every == 0 or k == args.steps:
            l7_save_state(outdir, st)
            emit_l7_candidate(best, outdir)
            if k % max(1, args.save_every * 2) == 0 or k == args.steps:
                print(f"[L7] step={st.step} best_mse={best_mse:.6g} ops={best_ops} policy={st.policy} saved={outdir}")

    print(f"[L7] done best_mse={best_mse:.6g} ops={best_ops} outdir={outdir.resolve()}")


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="RSI Unified One-File (SCIG / Ouroboros / L7)")
    ap.add_argument("--engine", choices=["scig", "ouroboros", "l7"], default="scig")

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--task", choices=["sin", "poly", "mix", "abs"], default="mix")
    ap.add_argument("--outdir", type=str, default="runs/unified")

    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=6)

    ap.add_argument("--w_size", type=float, default=None)
    ap.add_argument("--w_risk", type=float, default=None)
    ap.add_argument("--w_novelty", type=float, default=None)
    ap.add_argument("--holdout_slack", type=float, default=None)
    ap.add_argument("--stress_slack", type=float, default=None)
    ap.add_argument("--enable_self_ops", action="store_true", help="Enable self-inventing operators (SCIG).")
    ap.add_argument("--disable_self_ops", action="store_true", help="Disable self-inventing operators (SCIG).")
    ap.add_argument("--ops_synth_every", type=int, default=40, help="Synthesize/install a new operator every N steps (SCIG). 0 disables.")
    ap.add_argument("--ops_trials", type=int, default=6, help="Trials to estimate average gain of a synthesized operator.")
    ap.add_argument("--ops_min_avg_gain", type=float, default=1e-4, help="Minimum avg gain threshold to accept a synthesized operator.")
    ap.add_argument("--ops_lib_path", type=str, default="", help="Optional path to operators.json; default is <outdir>/operators.json")
    ap.add_argument("--self_patch_ops", action="store_true", help="If set, patch this source file OPERATORS_LIB block in-place when installing.")

    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--probe_steps", type=int, default=120)
    ap.add_argument("--emit_patched", type=str, default="")
    ap.add_argument("--emit_patched_inplace", action="store_true")

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--n_points", type=int, default=256)

    ap.add_argument("--domain_lo", type=float, default=-3.0)
    ap.add_argument("--domain_hi", type=float, default=3.0)
    return ap

def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.engine == "scig":
        tuned = dict(OUROBOROS_TUNED)
        enable_self_ops = True
        if getattr(args, "disable_self_ops", False):
            enable_self_ops = False
        if getattr(args, "enable_self_ops", False):
            enable_self_ops = True
        cfg = SCIGConfig(
            steps=args.steps,
            seed=args.seed,
            task=args.task,
            outdir=args.outdir,
            max_depth=args.max_depth if args.max_depth is not None else int(tuned["max_depth"]),
            w_size=args.w_size if args.w_size is not None else float(tuned["w_size"]),
            w_risk=args.w_risk if args.w_risk is not None else float(tuned["w_risk"]),
            w_novelty=args.w_novelty if args.w_novelty is not None else float(tuned["w_novelty"]),
            holdout_slack=args.holdout_slack if args.holdout_slack is not None else float(tuned["holdout_slack"]),
            stress_slack=args.stress_slack if args.stress_slack is not None else float(tuned["stress_slack"]),
enable_self_ops=enable_self_ops,
ops_synth_every=int(getattr(args, "ops_synth_every", 40)),
ops_trials=int(getattr(args, "ops_trials", 6)),
ops_min_avg_gain=float(getattr(args, "ops_min_avg_gain", 1e-4)),
ops_lib_path=str(getattr(args, "ops_lib_path", "")),
self_patch_ops=bool(getattr(args, "self_patch_ops", False)),
            domain_lo=args.domain_lo,
            domain_hi=args.domain_hi,
        )
        scig_run(cfg)
        return

    if args.engine == "ouroboros":
        emit = None
        if args.emit_patched_inplace:
            emit = str(Path(__file__).resolve())
        elif args.emit_patched and args.emit_patched.strip():
            emit = args.emit_patched.strip()
        args.emit_patched = emit
        ouroboros_run(args)
        return

    if args.engine == "l7":
        _require_numpy()
        l7_run(args)
        return

if __name__ == "__main__":
    main()
