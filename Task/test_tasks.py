"""Offline checks for the published task IDs; no benchmark installation is needed."""

import hashlib
import json
from itertools import combinations
from pathlib import Path, PurePosixPath

import pytest


ROOT = Path(__file__).resolve().parent
SET1 = {
    f"{benchmark}_tasks": (benchmark, count)
    for benchmark, count in {
        "appworld": 60,
        "browsecomp": 60,
        "swebench_lite": 100,
        "tau2": 100,
        "terminalbench": 60,
    }.items()
}
SCALE_UP = {
    **{
        f"appworld/appworld_{split}_set{number}_30": ("appworld", 30)
        for split in ("normal", "challenge")
        for number in (1, 2)
    },
    **{
        f"appworld/appworld_test_{split}_subset{number}_30": ("appworld", 30)
        for split in ("normal", "challenge")
        for number in (1, 2)
    },
    **{
        f"browsecomp/browsecomp_subset{number}_30": ("browsecomp", 30)
        for number in range(1, 6)
    },
    **{
        f"swebench/swebench_{repo}_{count}": ("swebench", count)
        for repo, count in (("django", 30), ("sympy", 30), ("matplotlib", 23), ("sklearn", 23))
    },
    **{
        f"tau2/tau2_{domain}_set{number}_30": ("tau2", 30)
        for domain in ("airline", "retail", "telecom")
        for number in (1, 2)
    },
    **{
        f"terminalbench/terminalbench_set{number}_30": ("terminalbench", 30)
        for number in (1, 2)
    },
}
COLLECTIONS = {"Set1_tasks": SET1, "scale_up_tasks": SCALE_UP}
REQUIRED_FIELDS = {"benchmark", "set_id", "n_tasks", "sha256_12", "task_ids"}
OPTIONAL_FIELDS = {
    "split", "domain", "repo", "dataset", "group_id", "selection_rule", "selection_group", "set_num",
    "start_index", "overlap_with_sibling_count", "overlap_with_sibling_task_ids",
    "source_index_range",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def task_ids(collection, stem):
    return read_json(ROOT / collection / f"{stem}.json")["task_ids"]


@pytest.mark.parametrize("collection", COLLECTIONS)
def test_manifest_covers_all_files_with_portable_paths(collection):
    directory = ROOT / collection
    manifest = read_json(directory / "MANIFEST.json")
    assert manifest["collection"] == collection
    entries = manifest["sets"]
    expected = COLLECTIONS[collection]
    assert len(entries) == len(expected)
    assert len({(entry["benchmark"], entry["set_id"]) for entry in entries}) == len(entries)

    for suffix in ("json", "txt"):
        references = [entry[suffix] for entry in entries]
        assert len(references) == len(set(references))
        assert set(references) == {f"{stem}.{suffix}" for stem in expected}
        for reference in references:
            path = PurePosixPath(reference)
            assert not path.is_absolute() and ".." not in path.parts
            assert "\\" not in reference and path.as_posix() == reference
            target = directory / reference
            assert target.is_file() and target.resolve().is_relative_to(directory.resolve())
        files = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob(f"*.{suffix}")
            if path != directory / "MANIFEST.json"
        }
        assert files == set(references)

    for entry in entries:
        data = read_json(directory / entry["json"])
        assert Path(entry["json"]).with_suffix(".txt").as_posix() == entry["txt"]
        for field in entry.keys() - {"json", "txt"}:
            assert entry[field] == data[field]


@pytest.mark.parametrize(
    "collection,stem,benchmark,count",
    [
        (collection, stem, benchmark, count)
        for collection, sets in COLLECTIONS.items()
        for stem, (benchmark, count) in sets.items()
    ],
)
def test_task_records_preserve_order_and_digest(collection, stem, benchmark, count):
    directory = ROOT / collection
    data = read_json(directory / f"{stem}.json")
    assert REQUIRED_FIELDS <= data.keys() <= REQUIRED_FIELDS | OPTIONAL_FIELDS
    assert data["benchmark"] == benchmark
    assert isinstance(data["set_id"], str) and data["set_id"].strip()
    ids = data["task_ids"]
    assert isinstance(ids, list) and len(ids) == data["n_tasks"] == count
    assert all(isinstance(value, str) and value and value == value.strip() for value in ids)
    assert len(set(ids)) == count
    ordered = "\n".join(ids).encode("utf-8")
    assert (directory / f"{stem}.txt").read_bytes() == ordered + b"\n"
    assert hashlib.sha256(ordered).hexdigest()[:12] == data["sha256_12"]


@pytest.mark.parametrize("domain,expected_count", [("airline", 10), ("retail", 0), ("telecom", 0)])
def test_tau2_sibling_overlap_is_explicit(domain, expected_count):
    siblings = [
        read_json(ROOT / "scale_up_tasks" / "tau2" / f"tau2_{domain}_set{number}_30.json")
        for number in (1, 2)
    ]
    overlap = set(siblings[0]["task_ids"]) & set(siblings[1]["task_ids"])
    assert len(overlap) == expected_count
    for sibling in siblings:
        assert sibling["domain"] == domain
        assert sibling["overlap_with_sibling_count"] == expected_count
        assert sibling["overlap_with_sibling_task_ids"] == [
            value for value in sibling["task_ids"] if value in overlap
        ]


@pytest.mark.parametrize("benchmark,prefix", [("appworld", "normal_"), ("terminalbench", "")])
def test_canonical_halves_reconstruct_set1(benchmark, prefix):
    halves = [
        task_ids("scale_up_tasks", f"{benchmark}/{benchmark}_{prefix}set{number}_30")
        for number in (1, 2)
    ]
    assert not set(halves[0]) & set(halves[1])
    assert halves[0] + halves[1] == task_ids("Set1_tasks", f"{benchmark}_tasks")


def test_browsecomp_scale_up_subsets_are_disjoint_from_each_other_and_set1():
    subsets = [set(task_ids("Set1_tasks", "browsecomp_tasks"))] + [
        set(task_ids("scale_up_tasks", f"browsecomp/browsecomp_subset{number}_30"))
        for number in range(1, 6)
    ]
    for left, right in combinations(subsets, 2):
        assert not left & right


def test_appworld_canonical_and_additional_subsets_remain_distinct():
    records = [
        read_json(path)
        for path in (ROOT / "scale_up_tasks" / "appworld").glob("*.json")
    ]
    assert len(records) == 8
    for record in records:
        additional = "_subset" in record["set_id"]
        assert record["selection_group"] == (
            "additional_random" if additional else "canonical"
        )
        assert record["split"] == (
            "test_challenge" if "challenge" in record["set_id"] else "test_normal"
        )
    for left, right in combinations(records, 2):
        assert not set(left["task_ids"]) & set(right["task_ids"])
