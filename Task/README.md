# Task selections

Fixed, ordered task-ID selections used by the experiments. The two selection
folders below hold task lists. Execution adapters, required assets, and local
data preparation commands are also provided under `Task/`; see
[Running the task selections](RUNNING.md) to load and execute the actual tasks.

```text
Task/
  Set1_tasks/          Initial five benchmark selections
    MANIFEST.json
    *_tasks.json
    *_tasks.txt
  scale_up_tasks/      Scale-up selections, grouped by benchmark
    MANIFEST.json
    appworld/
    browsecomp/
    swebench/
    tau2/
    terminalbench/
```

| Benchmark | Set1 tasks | Scale-up selections | Scale-up task entries |
| --- | ---: | --- | ---: |
| AppWorld | 60 | 4 canonical sets + 4 additional random subsets, each with 30 tasks | 240 |
| BrowseComp | 60 | 5 sets of 30 | 150 |
| SWE-bench Lite | 100 | Django 30, SymPy 30, Matplotlib 23, scikit-learn 23 | 106 |
| tau2 | 100 | 2 sets of 30 for each of airline, retail, and telecom | 180 |
| Terminal-Bench | 60 | 2 sets of 30 | 60 |
| Total | 380 | 25 selections | 736 |

Counts are task entries, not globally unique tasks. Scale-up contains 726
unique benchmark-qualified IDs, including the additional AppWorld subsets.

## File format

Each selection has a JSON object and a matching TXT file. JSON uses the same
schema across benchmarks:

- `benchmark`, `set_id`: identify the selection.
- `n_tasks`, `sha256_12`: size and ordered-ID checksum.
- `task_ids`: ordered strings, preserved exactly from the original selection.
- Optional `split`, `domain`, `dataset`, `repo`, and selection metadata describe
  the benchmark and sampling procedure.

TXT contains one task ID per line. Each collection's `MANIFEST.json` indexes
every JSON/TXT pair, with paths relative to that manifest's directory.
The checksum is the first 12 hexadecimal characters of
`sha256("\n".join(task_ids).encode("utf-8"))`, without a trailing newline.

```python
import json
from pathlib import Path

root = Path("Task/scale_up_tasks")
manifest = json.loads((root / "MANIFEST.json").read_text())
selection = json.loads((root / manifest["sets"][0]["json"]).read_text())
task_ids = selection["task_ids"]
```

## Selection relationships

- **AppWorld:** the four `normal_set*` / `challenge_set*` files are the
  canonical scale-up partitions recorded in the original experiment manifest.
  The normal pair partitions the Set1 `test_normal` selection; the challenge
  pair partitions the original Set2 `test_challenge` selection. The four
  `test_*_subset*` files are separate random samples, selected with seed
  `20260516` after excluding prior same-split tasks. They were present in the
  original task directory but absent from its experiment manifest. All eight
  are retained, with `selection_group` set to `canonical` or
  `additional_random`.
- **tau2:** airline has 50 base tasks. Its two 30-task windows deliberately
  overlap on 10 IDs; retail and telecom windows are disjoint. The JSON files
  retain the window offsets and sibling-overlap metadata.
- **Terminal-Bench:** the two scale-up selections partition the initial
  60-task Set1 sequence in order.
- **BrowseComp:** the five scale-up selections are mutually disjoint and
  exclude the initial Set1 selection. Source-index ranges are retained.

These collections are experimental selections, not globally disjoint splits.
Several scale-up selections reuse Set1 tasks. Original filenames and TXT
contents are preserved; JSON metadata is normalized and excludes run
configuration, local paths, question excerpts, and answer fields.

## Offline validation

With Python 3.11+ and pytest installed, run from the repository root:

```bash
python -m pytest -q Task/test_tasks.py
```

Tests check complete manifest coverage, portable paths, ordered JSON/TXT
agreement, checksums, counts, within-set uniqueness, and known cross-set
relationships. They do not download datasets or execute benchmark tasks.
