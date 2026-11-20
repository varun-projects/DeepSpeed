# Glossary and code-reading map

**What this covers.** The vocabulary you need before the other documents in this folder make
sense, and a map from each term to the file that defines it. DeepSpeed's source uses a lot of
short, internal names (`ds_tensor`, `ipg`, `mpu`, `uc`); nearly every one of them is defined in
exactly one place, listed here.

**Source of truth.** Upstream commit `d70adc32`, DeepSpeed version `0.20.1`. Every entry below was
checked against that tree.

---

## How to read this folder

| Document | Read it when |
| --- | --- |
| `01-architecture.md` | You need the shape of the whole system and who owns what. |
| `02-engine-lifecycle.md` | You are tracing a single training step or an `initialize()` problem. |
| `03-zero.md` | You are touching sharding, gradient reduction, or parameter gathering. |
| `04-offload-and-memory.md` | You are chasing memory: offload, NVMe, activation checkpointing. |
| `05-parallelism.md` | You are working on pipeline, tensor, or sequence parallelism. |
| `06-moe-and-expert-parallelism.md` | You are working on MoE, expert parallelism, or AutoEP. |
| `07-checkpointing.md` | You are touching save/load, universal checkpoints, or reshaping. |
| `08-inference.md` | You are working on the inference engine, injection, or kernels. |
| `09-configuration.md` | You are adding or debugging a config key, or the batch-size math. |
| `10-comm-and-accelerators.md` | You are adding a collective, or porting to an accelerator. |
| `11-deepcompile.md` | You are working on the compile subsystem. |
| `12-testing-and-ci.md` | You need to run or add tests, or understand CI selection. |
| `case-studies/` | You want worked examples of real defects and how they were fixed. |

---

## Core nouns

### Engine

**`DeepSpeedEngine`** — `deepspeed/runtime/engine.py`. The `nn.Module` wrapper returned by
`deepspeed.initialize()`. Owns the config, the wrapped model, the optimizer wrapper, the LR
scheduler, communication groups, and checkpoint save/load. Almost every training-path bug lands
either here or in a ZeRO optimizer.

**`PipelineEngine`** — `deepspeed/runtime/pipe/engine.py`. Subclass used when the model is a
`PipelineModule`. Replaces `forward`/`backward`/`step` with an instruction schedule, so a fix made
in `DeepSpeedEngine` does not automatically apply to pipeline runs.

**`InferenceEngine`** — `deepspeed/inference/engine.py`. Returned by `deepspeed.init_inference()`.
A separate object from the training engine; they share the accelerator and op layers but not the
step logic.

**Hybrid engine** — `deepspeed/runtime/hybrid_engine.py`. A training engine that can switch into
an inference-style layout for generation, used by RLHF-style loops.

### Config

**`DeepSpeedConfig`** — `deepspeed/runtime/config.py`. The parsed, validated config. Accepts a
dict, a filesystem path, or a base64 blob. Holds `world_size`, the batch triple, and the
sub-config objects.

**`DeepSpeedConfigModel`** — `deepspeed/runtime/config_utils.py`. The pydantic base class that
newer config blocks derive from, with deprecation handling.

**The batch triple** — `train_batch_size`, `train_micro_batch_size_per_gpu`,
`gradient_accumulation_steps`. The invariant, asserted in `_batch_assertion`
(`deepspeed/runtime/config.py:606`), is
`train_batch_size == train_micro_batch_size_per_gpu * gradient_accumulation_steps * world_size`.
Supply any two and DeepSpeed derives the third. Note that `world_size` here means the *data
parallel* degree, not necessarily `dist.get_world_size()`.

### ZeRO

**ZeRO stage** — how much optimizer-related state is sharded across data-parallel ranks.
Stage 1 shards optimizer state, stage 2 adds gradients, stage 3 adds the parameters themselves.
Stages 1 and 2 live in `deepspeed/runtime/zero/stage_1_and_2.py`; stage 3 lives in
`deepspeed/runtime/zero/stage3.py`. These are two separate implementations, which is why a bug
found in one is worth checking in the other.

**`ds_tensor`** — on a stage-3 parameter, the rank's local shard of that parameter. The full
tensor only exists while gathered. Set up in `deepspeed/runtime/zero/partition_parameters.py`.

**`ds_id`** — the unique integer DeepSpeed assigns each stage-3 parameter, used as the key in
coordination and tracing structures.

**`ds_numel` / `ds_shape`** — the *unpartitioned* number of elements and shape of a stage-3
parameter, retained because `param.shape` reflects the local shard while partitioned.

**`ZeroParamStatus`** — `deepspeed/runtime/zero/partition_parameters.py:237`. Enum with
`AVAILABLE`, `NOT_AVAILABLE`, and an in-flight state; the stage-3 parameter state machine.

**`ZeroParamType`** — `deepspeed/runtime/zero/partition_parameters.py:224`. `NORMAL`,
`PARTITIONED`, or held by a unique rank.

**`zero.Init`** — the context manager in `deepspeed/runtime/zero/partition_parameters.py` that
intercepts module construction so that parameters are created already partitioned. Needed for
models too large to instantiate on one device.

**`GatheredParameters`** — `deepspeed/runtime/zero/partition_parameters.py`. Context manager that
temporarily all-gathers stage-3 parameters so ordinary code can read or write full tensors.

**`ZeROOrderedDict`** — `deepspeed/runtime/zero/parameter_offload.py`. The parameter dict
substituted into modules so that attribute access can trigger a gather.

**IPG** — "independent parallel gradient". The bucketing mechanism that batches gradients into a
contiguous buffer before reducing them, so communication is a few large collectives rather than
one per parameter. See `ipg_bucket` handling in `deepspeed/runtime/zero/stage_1_and_2.py` and
`deepspeed/runtime/base_optimizer.py`.

**`grad_accum` / `use_grad_accum_attribute`** — the separate accumulation buffer used when the
gradient accumulation dtype differs from the parameter dtype, so a window of micro-batch
gradients can be summed at higher precision than the parameters.

**Persistent parameters** — small stage-3 parameters kept permanently gathered because sharding
them costs more in communication than it saves in memory
(`stage3_param_persistence_threshold`).

### Parallelism

**`mpu`** — "model parallel unit". The object (historically from Megatron-LM) that answers
`get_data_parallel_world_size()`, `get_model_parallel_rank()`, and similar. DeepSpeed accepts one
in `deepspeed.initialize()` and uses it to learn the data-parallel degree; see
`deepspeed/runtime/config.py:460`.

**DP / TP / PP / EP / SP** — data, tensor, pipeline, expert, and sequence parallelism. The process
groups for each are created and looked up through `deepspeed/utils/groups.py`.

**`ProcessTopology`** — `deepspeed/runtime/pipe/topology.py`. Maps a rank to its coordinates in the
parallelism grid.

**AutoTP** — automatic tensor parallelism: match known module patterns and replace them with
sharded equivalents, instead of requiring a hand-written parallel model.
`deepspeed/module_inject/auto_tp.py`, `deepspeed/runtime/tensor_parallel/`.

**Ulysses / sequence parallelism** — shards the sequence dimension across ranks and uses all-to-all
to switch between a sequence-sharded and a head-sharded layout inside attention. Two
implementations exist in this tree: the older `deepspeed/sequence/layer.py` (`DistributedAttention`)
and the newer `deepspeed/runtime/sequence_parallel/ulysses_sp.py`.

**MoE / expert parallelism** — `deepspeed/moe/`. A gate routes each token to one or more experts;
experts are distributed across ranks, so routing is an all-to-all. `capacity_factor` and
`drop_tokens` decide what happens when an expert is oversubscribed.

**AutoEP** — the newer automatic expert-parallel path, `deepspeed/module_inject/auto_ep_layer.py`,
which can fold together with AutoTP.

### Checkpointing

**Model states vs optimizer states** — a DeepSpeed checkpoint tag directory holds
`*_model_states.pt` (module tensors and metadata) and `*_optim_states.pt` (optimizer/ZeRO shards).

**`latest` / `latest_universal`** — the small text files in the checkpoint root naming the tag to
load. `latest_universal` is written for converted universal checkpoints; see
`deepspeed/checkpoint/ds_to_universal.py` and `deepspeed/runtime/engine.py:4550`.

**UCP, universal checkpoint** — a parallelism-independent checkpoint format: one directory per
parameter under `zero/`, holding `fp32.pt` and the optimizer moments, so a run saved at one
parallel layout can resume at another. Converted offline by
`deepspeed/checkpoint/ds_to_universal.py`; consumed when `checkpoint: {load_universal: true}`.

**`frozen_param_shapes` / `frozen_param_fragments`** — `deepspeed/checkpoint/constants.py:34-35`.
Where the save path records parameters with `requires_grad=False`, which the optimizer does not
own and which therefore have no fp32 master copy.

**`buffer_names`** — `deepspeed/checkpoint/constants.py:33`. The registered buffers (for example
BatchNorm `running_mean`) recorded alongside the module state, for the same reason.

**`DeepSpeedCheckpoint`** — `deepspeed/checkpoint/deepspeed_checkpoint.py`. The reader used by the
conversion and reshaping tools; parses `layer_<n>-model_<m>-model_states.pt` filenames via
`LAYER_FILE_PREFIX_PATTERN`.

### Platform

**Accelerator** — the device abstraction. `accelerator/abstract_accelerator.py` defines the
contract; `accelerator/real_accelerator.py` picks a concrete backend at import time. Device-neutral
code calls `get_accelerator()` rather than `torch.cuda`.

**`deepspeed.comm`** — `deepspeed/comm/`. DeepSpeed's wrapper over the distributed backend, which
adds timing, logging, and backend selection. Project rule: never `import torch.distributed`
directly, always `import deepspeed.comm as dist`. A pre-commit hook (`check-torchdist`) enforces
this.

**Op builder** — `op_builder/`. Each compiled extension has a builder class that reports whether it
`is_compatible()` with the current machine and can `load()` it, either from a prebuilt wheel or by
JIT-compiling the sources under `csrc/`. `DS_BUILD_*` environment variables force ahead-of-time
builds at install time.

**`ds_report`** — `deepspeed/env_report.py`. Prints which ops are installed and compatible; the
first thing to ask for in a bug report.

---

## Naming conventions worth knowing

| Pattern | Meaning |
| --- | --- |
| `_` prefix on a method | Internal; no stability promise, but widely called across the package. |
| `ds_*` attribute on a tensor or parameter | DeepSpeed metadata attached to a torch object. |
| `*_optim_states.pt`, `*_model_states.pt` | Checkpoint shard files. |
| `zero_pp_rank_<dp>_mp_rank_<mp>_*` | A checkpoint shard's position in the parallel grid. |
| `ep_size_<n>` | Expert-parallel group name, used as a key for MoE state. |
| `tests/unit/v1/` | The subset of unit tests that the GPU `modal-torch-latest` workflow runs. |

---

## Invariants that are easy to break

These come from `AGENTS.md` and from the code itself. All of them are enforced by a hook or a test,
so breaking one shows up in CI rather than in review.

1. **No direct `torch.distributed` import.** Use `deepspeed.comm`. Enforced by the `check-torchdist`
   pre-commit hook.
2. **`import deepspeed` must not initialize a CUDA context,** and must stay fast. This is what makes
   `pytest --forked` safe; see the note in `CONTRIBUTING.md`.
3. **Device neutrality.** Reach for `get_accelerator()`, not `torch.cuda`, in runtime code.
4. **New source files need the license header** (`# SPDX-License-Identifier: Apache-2.0` /
   `# DeepSpeed Team`). Enforced by the `check-license` hook.
5. **No cosmetic-only edits to existing code.** Diff noise is treated as a review cost.
6. **Parallel implementations must be checked together.** ZeRO 1/2 vs 3, fp16 vs bf16 optimizers,
   and the per-accelerator op builders are independent code paths that frequently share a defect.
