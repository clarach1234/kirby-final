# Team kirby — GAVE2 preliminary final submission

This repository is the **final-submission-only** code record for team
**kirby**'s highest-scoring GAVE2 preliminary artifact. It deliberately
excludes earlier experiments, challenge data, weights, prediction images, and
submission archives.

| Official preliminary result | Value |
|---|---:|
| Rank | 7 |
| Overall score | 7.95236 |
| Task 1 / Task 2 / Task 3 | 8.19815 / 8.25683 / 7.52500 |
| Scored ZIP SHA-256 | `4c19a97503668ef003c992edcbf76ade644c40903b683394347c0214db51306c` |
| ZIP size | 19,178,103 bytes |

Start here: [`final_submission/README.md`](final_submission/README.md).

## Scope and reproducibility

The repository contains the selected training/inference source code, fixed
post-processing, Dockerfiles, and SHA-256 manifests. The official data and
local checkpoints are supplied only through a restricted organizer link and
mounted read-only at runtime. The public R2-V2 AV checkpoint is not
redistributed; [`material_review/fetch_r2v2.sh`](material_review/fetch_r2v2.sh)
obtains the pinned upstream release and verifies its hash.

The primary raw-input command is:

```bash
bash material_review/run_end_to_end_docker.sh DATA_DIR WEIGHTS_DIR OUTPUT_DIR
```

It runs both required Docker images, regenerates all Task 1/2/3 outputs, and
requires the final ZIP to match the scored SHA-256 above. See
[`final_submission/DOCKER.md`](final_submission/DOCKER.md) for the expected
directory layout and the independently completed GPU test.

## Attribution

The local implementation is based on CMRRWNet and RRWNet under the included
MIT License. Task 1 uses the separately fetched public R2-V2 AV specialist;
the source commit and release hashes are recorded in
[`material_review/weights_manifest.json`](material_review/weights_manifest.json).
