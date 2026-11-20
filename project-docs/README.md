# DeepSpeed, explained

This folder is a from-source technical guide to the DeepSpeed codebase: what the system is made
of, how a training step actually executes, how each major feature is implemented, and a set of
worked case studies of real defects that were found and fixed in this repository.

It was written by reading the code, not the marketing material. Every non-obvious claim carries a
`path/file.py:LINE` citation so you can check it. The diagrams are Mermaid, so they render inline
on GitHub.

**Source of truth.** Upstream commit `d70adc32` (`deepspeedai/DeepSpeed` `master`), which
`version.txt` labels `0.20.1`. The tree these documents describe is byte-identical to upstream
`master` at that commit — this fork adds no source changes, only the `project-docs/` folder you
are reading. Note that `master` is *ahead* of the published `0.20.1` release, so a few subsystems
described here (ZenFlow, SuperOffload, AutoEP, the affine-transfer checkpoint IR, DeepCompile
passes) are post-release code that a released wheel may not contain; where that matters, the text
says so. Line numbers drift; the surrounding function and class names are the durable part of a
citation.

---

## Start here

If you are new to the codebase, read in this order:

1. **[01-architecture.md](01-architecture.md)** — the component map: subpackages, layering, the
   accelerator abstraction, and how the parallelism dimensions compose.
2. **[02-engine-lifecycle.md](02-engine-lifecycle.md)** — `deepspeed.initialize()` through one
   full training step, including the gradient-accumulation boundary.
3. **[13-glossary.md](13-glossary.md)** — the internal vocabulary (`ds_tensor`, `ipg`, `mpu`,
   `uc`) and the invariants that are easy to break.

Then go wherever your problem lives.

## Feature documents

| Document | Subsystem | Primary source |
| --- | --- | --- |
| [01-architecture.md](01-architecture.md) | Whole-system structure | `deepspeed/`, `accelerator/`, `op_builder/` |
| [02-engine-lifecycle.md](02-engine-lifecycle.md) | Engine construction and the training step | `deepspeed/runtime/engine.py` |
| [03-zero.md](03-zero.md) | ZeRO stages 1, 2 and 3 | `deepspeed/runtime/zero/` |
| [04-offload-and-memory.md](04-offload-and-memory.md) | CPU/NVMe offload, activation checkpointing | `deepspeed/runtime/zero/`, `swap_tensor/`, `activation_checkpointing/` |
| [05-parallelism.md](05-parallelism.md) | Pipeline, tensor and sequence parallelism | `deepspeed/runtime/pipe/`, `tensor_parallel/`, `sequence/` |
| [06-moe-and-expert-parallelism.md](06-moe-and-expert-parallelism.md) | MoE, expert parallelism, AutoEP | `deepspeed/moe/`, `module_inject/auto_ep_layer.py` |
| [07-checkpointing.md](07-checkpointing.md) | Save/load, universal checkpoints, reshaping | `deepspeed/checkpoint/`, `runtime/engine.py` |
| [08-inference.md](08-inference.md) | Inference engine, injection, kernels | `deepspeed/inference/`, `module_inject/`, `ops/transformer/` |
| [09-configuration.md](09-configuration.md) | Config parsing, validation, batch math | `deepspeed/runtime/config.py` |
| [10-comm-and-accelerators.md](10-comm-and-accelerators.md) | Collectives, accelerators, op builders | `deepspeed/comm/`, `accelerator/`, `op_builder/` |
| [11-deepcompile.md](11-deepcompile.md) | The compile subsystem | `deepspeed/compile/` |
| [12-testing-and-ci.md](12-testing-and-ci.md) | Tests, CI workflows, diff-based selection | `tests/`, `ci/`, `.github/workflows/` |
| [13-glossary.md](13-glossary.md) | Terminology and conventions | — |

## Case studies

**[case-studies/](case-studies/)** holds deep dives into real, already-fixed defects in this
repository — chosen for being instructive rather than dramatic. Each one follows the same shape:
symptom, the background mechanics you need, the precise root cause, a walk through the actual
diff, why the obvious alternative fix is wrong, how it was verified, and the generalisable lesson.
Each has flowcharts of the buggy and the fixed control flow.

See **[case-studies/README.md](case-studies/README.md)** for the index.

## Finding your way around the source

A map from "I need to change X" to the file that owns it:

| I need to change... | Look in |
| --- | --- |
| How a training step is sequenced | `deepspeed/runtime/engine.py` |
| Gradient reduction for ZeRO 1/2 | `deepspeed/runtime/zero/stage_1_and_2.py` |
| Parameter gathering for ZeRO 3 | `deepspeed/runtime/zero/stage3.py`, `partition_parameters.py`, `partitioned_param_coordinator.py` |
| A config key's parsing or default | `deepspeed/runtime/config.py`, the per-feature `config.py`, `constants.py` |
| Checkpoint format or conversion | `deepspeed/checkpoint/`, `deepspeed/utils/zero_to_fp32.py` |
| A collective call | `deepspeed/comm/comm.py` |
| Device-specific behaviour | `accelerator/*_accelerator.py` |
| A compiled kernel | `csrc/`, with its builder in `op_builder/` |
| Pipeline scheduling | `deepspeed/runtime/pipe/schedule.py`, `engine.py` |
| MoE routing | `deepspeed/moe/sharded_moe.py` |
| Inference module replacement | `deepspeed/module_inject/replace_module.py`, `auto_tp.py` |
| Test helpers for distributed tests | `tests/unit/common.py`, `tests/unit/util.py` |

## Conventions used in this folder

- **Citations** look like `deepspeed/runtime/engine.py:4780`. They point at commit `d70adc32`.
- **Diagrams** are Mermaid in fenced ` ```mermaid ` blocks. A red-tinted node in a case-study
  diagram marks the defect point.
- **Tables** are preferred over prose for anything enumerable (config keys, per-stage behaviour).
- Where something could not be verified from the source alone, the text says so explicitly rather
  than guessing. An unmarked claim is one that was checked.

## Related documents in the repository root

These are the project's own rules, and they take precedence over anything written here:

- `CONTRIBUTING.md` — pre-commit setup, how to run tests, DCO, the new-feature proposal process.
- `AGENTS.md` — mandatory rules for AI-assisted contributions, plus the project's code and test
  discipline rules.
- `.agents/skills/pr-checklist/SKILL.md` — the checklist to work through before asking for review.
- `.github/workflows/TEST_SELECTION.md` — how diff-based GPU test selection works.
