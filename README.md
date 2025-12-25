# RSI Operator Synthesis

## Overview
Self-improving symbolic regression engine that synthesizes new search operators through meta-learning.

## Core Features
- **Operator Synthesis**: Automatically creates composite mutation operators from primitives
- **Meta-Learning**: Evaluates and retains effective operator combinations
- **Self-Modification**: Optionally patches discovered operators back into source code (`--self_patch_ops`)

## Technical Details
- **Base Algorithm**: Evolutionary symbolic regression (genetic programming)
- **Search Space**: Mathematical expressions (sin, sqrt, log, +, *, etc.)
- **Meta-Operators**: Combinations of `simplify`, `mut_const`, `insert_binary`, `swap_subtrees`, etc.
- **Evaluation**: Multi-objective (MSE, complexity, novelty)

## Verified Results
- **Test Duration**: 2+ hours, 1.3M+ iterations
- **Operators Created**: 55+ novel composite operators
- **Performance**: Objective score improved ~10x (2.82 → 0.27)
- **Acceptance Rate**: ~0.002% (highly selective)

## Usage
```bash
python rsi_unified_onefile.py --engine scig --steps 10000 --task mix --outdir runs/test
```

### Key Flags
- `--ops_synth_every N`: Synthesize operator every N steps
- `--ops_trials K`: Evaluation trials per candidate operator
- `--ops_min_avg_gain G`: Minimum gain threshold for acceptance
- `--self_patch_ops`: Enable source code self-modification

## Scope
This is a **domain-specific self-improvement system** for symbolic regression, not general AI. It improves its own search strategy within fixed algorithmic constraints.
