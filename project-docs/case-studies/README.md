# DeepSpeed engineering case studies

In-depth write-ups of real, already-fixed DeepSpeed bugs. Each one starts from the bug report
as it was filed, follows the mechanics down to the exact condition that triggered the failure,
quotes the fix as it landed, and ends with a snippet you can run on a CPU-only box to see the
current behaviour for yourself.

These are documentation, not tests. Nothing here is imported by the package and no snippet in
these files is checked in as a test file.

## The seven case studies

| Id | Title | Issue / PR | Subsystem | Difficulty | One-line summary |
|---|---|---|---|---|---|
| [cs-01](cs-01-engine-zero-memory-leak.md) | A destroyed DeepSpeed engine that nothing could reclaim | issue [#7204](https://github.com/deepspeedai/DeepSpeed/issues/7204), PR [#9034](https://github.com/deepspeedai/DeepSpeed/pull/9034) | engine / ZeRO teardown | Difficult | Seven retaining references across five subsystems kept a torn-down engine and its model alive, so re-initializing in one process leaked the whole object graph. |
| [cs-02](cs-02-zero12-cpu-gradient-bucket-drop.md) | The all-reduce that was skipped because a string did not match | PR [#8382](https://github.com/deepspeedai/DeepSpeed/pull/8382) (surfaced by [#8381](https://github.com/deepspeedai/DeepSpeed/pull/8381)) | ZeRO-1/2 gradient reduction | Difficult | A device-name string comparison sent the non-contiguous CPU reduction path down the wrong branch, silently dropping a gradient bucket in a matrix cell CI never ran. |
| [cs-03](cs-03-zero3-muon-allgather-mixed-shapes.md) | A round-robin all-gather that assumed every parameter had the same shape | PR [#8628](https://github.com/deepspeedai/DeepSpeed/pull/8628) | ZeRO-3 + Muon optimizer | Difficult | `all_gather` with differently shaped outputs is tolerated by NCCL and rejected by gloo, so the Muon all-gather was correct on GPU and broken on CPU for the same model. |
| [cs-04](cs-04-checkpoint-layer-key-prefix-collision.md) | Checkpoint layer keys collide by prefix at 100+ pipeline layers | issue [#3305](https://github.com/deepspeedai/DeepSpeed/issues/3305) (open), fixed by PR [#5506](https://github.com/deepspeedai/DeepSpeed/pull/5506) | checkpoint reader / reshaping | Medium | `fname[:len('layer_') + 2]` folded `layer_100` onto `layer_10`, so a 120-layer checkpoint read back as 100 layers with 11x oversized per-layer file groups. |
| [cs-05](cs-05-flops-profiler-patch-leak.md) | The flops profiler leaked its own monkeypatches | issue [#2231](https://github.com/deepspeedai/DeepSpeed/issues/2231) (open), fixed by PRs [#2068](https://github.com/deepspeedai/DeepSpeed/pull/2068) and [#8246](https://github.com/deepspeedai/DeepSpeed/pull/8246) | flops profiler | Medium | The patch list and the restore list were hand-maintained and unequal, so repeated profiling wrapped the wrappers and reported doubling flops for the same model. |
| [cs-06](cs-06-flops-profiler-upsample-scale-factor.md) | Upsample flops used the batch size as an exponent | issue [#5537](https://github.com/deepspeedai/DeepSpeed/issues/5537) (open), fixed by PR [#7353](https://github.com/deepspeedai/DeepSpeed/pull/7353) | flops profiler | Medium | `scale_factor ** len(input)` raised the scale to the batch size instead of the spatial-dim count, reporting 6291456 flops where 98304 was right. |
| [cs-07](cs-07-zero12-zero-sized-parameters.md) | ZeRO-1/2 KeyError on a trailing zero-sized parameter | issue [#8279](https://github.com/deepspeedai/DeepSpeed/issues/8279), PR [#8280](https://github.com/deepspeedai/DeepSpeed/pull/8280) | ZeRO-1/2 gradient reduction | Medium | A zero-element parameter at the end of a group matched neither interval branch in the partition walk, so it was bucketed for reduction but absent from `param_to_partition_ids`. |

Three of the four Medium studies (cs-04, cs-05, cs-06) share a trait worth knowing about before
you read them: **the issue is still open even though master is fixed**. In each case the repair
shipped inside a PR that referenced a different ticket, or no ticket at all. Verifying that the
defect is really gone from the current tree -- and reporting what is *not* gone -- is part of
each write-up rather than something taken on trust.

## How to read a case study

Every file follows the same seven-part structure, so you can enter at whichever part you need:

1. **Title and fact box** -- issue and PR numbers with dates and state, the files involved, the
   difficulty, and a one-line "why this is instructive". If an issue is still open while the
   code is fixed, the fact box says so explicitly.
2. **Symptom** -- short quotations from the report. What the user saw, in their words, including
   the traceback or the wrong number. No analysis yet.
3. **Background** -- only the mechanics you need to follow the rest, every claim cited into the
   current tree as `path/file.py:LINE`. Line numbers refer to this checkout (DeepSpeed 0.20.1,
   branch `intro`, based on upstream `d70adc32`) and will drift against upstream master.
4. **Root cause** -- the exact condition that triggers the failure, plus a worked example with
   real numbers or real filenames. If you only read one part, read this one.
5. **The fix** -- the current code as a fenced block and a `diff`-style before/after. Each study
   states where its "before" came from: a merged upstream diff, or a reconstruction from the
   issue text. Where a sibling defect survived the fix, it is called out under a
   "residual risk" heading with measurements, not adjectives.
6. **How to verify it yourself** -- a short CPU-only snippet and the *real* output it produced on
   this tree. Nothing in this section is aspirational; if a snippet needed an environment
   workaround (no C++ compiler, no GPU), the workaround is in the snippet and explained.
7. **Lessons** -- three to five bullets, aimed at the next person writing code in that
   subsystem.

Two Mermaid flowcharts close each file: the buggy flow, with the defect node styled in red, and
the fixed flow. They are a map of the control path, not a replacement for the root-cause
section.

## The method

The research path for every study is the same four steps, in this order:

1. **Issue.** Fetch the report and all comments through the GitHub API
   (`/repos/deepspeedai/DeepSpeed/issues/<n>` and `.../comments`). The reporter's reproducer and
   their *controls* ("stage 0 passes, DDP passes, one GPU still fails") usually bound the search
   space more tightly than any amount of code reading.
2. **Root cause in the current tree.** Read the code the issue points at, then follow the data:
   who writes the structure, who reads it, and under which condition each one runs. Prefer
   `grep` over memory for sibling call sites -- in cs-04 and cs-07 the sibling is where the
   remaining risk lives.
3. **Diff.** Find the commit or PR that actually changed the behaviour
   (`/repos/.../commits?path=<file>` to list the history of one file, then the diff media type
   `Accept: application/vnd.github.v3.diff` on the commit or PR). The fix is frequently *not*
   the PR the issue links to; cs-04, cs-05 and cs-06 were all repaired by PRs that never
   mention the ticket. Quote the real `+`/`-` lines rather than paraphrasing them.
4. **Verification.** Re-run the symptom against the current tree, on CPU, and record the real
   output. Where the historical code is short enough, keep a local copy of the old body next to
   the current one and print both answers against an independent oracle -- the output size of
   `F.interpolate` in cs-06, the file list itself in cs-04. A study that cannot measure the
   difference says so.

Running the snippets in this folder:

```bash
cd /workshop/deepspeed/repo
PYTHONPATH=. DS_ACCELERATOR=cpu python3.13 /tmp/snippet.py
```

Keep the snippets outside the repository. They are illustrations for a reader, not part of the
test suite; the project's test discipline (see `AGENTS.md`) is a different bar, and the real
regression tests for these bugs live under `tests/unit/`.

## Conventions

* Line references are `path/file.py:LINE` against this checkout, in the form the editor and
  `grep -n` both produce.
* Issue and PR numbers are always linked on first use in a file.
* Numbers in the text are measured, not estimated. Where a number came from a PR description or
  an issue comment rather than from a local run, the text says whose number it is.
* "Residual risk" means: still wrong, or still reachable, on this tree today -- with the
  measurement that shows it.
