"""Writers and self-checks for the pipeline outputs (official competition formats)."""

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from src.blocking import CandidateGenerator
from src.data_loading import CANDIDATE_COLUMNS
from src.postprocessing.one_owner import verify_one_owner

CANDIDATE_PAIRS_HEADER = "source1_entity_id\tcandidate_entity_ids"
MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids"


def write_candidate_pairs(cand: pd.DataFrame, s1_ids: Iterable[str], path: Path) -> Path:
    """Official candidate_pairs.tsv (`source1_entity_id \\t candidate_entity_ids`), reusing P2's exporter."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ordered = cand.sort_values(["s1_id", "retrieval_rank"], kind="stable")
    return Path(CandidateGenerator.export_candidate_pairs_tsv(ordered, list(s1_ids), str(path)))


def write_candidates_detailed(cand: pd.DataFrame, path: Path) -> Path:
    """Internal 7-column candidate representation as TSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cand[CANDIDATE_COLUMNS].to_csv(path, sep="\t", index=False)
    return Path(path)


def write_matching_results(matches: pd.DataFrame, s1_ids: Iterable[str], path: Path, score_col: str = "score") -> Path:
    """
    Official matching_results.tsv: one row per S1 entity, comma-separated matched S2/S3 IDs (empty for no match),
    IDs ordered by descending score.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    by_s1 = {}
    if len(matches):
        m = matches.sort_values(["s1_id", score_col, "source_record_id"], ascending=[True, False, True], kind="mergesort")
        by_s1 = m.groupby("s1_id", sort=False)["source_record_id"].apply(list).to_dict()
    with open(path, "w", encoding="utf-8") as f:
        f.write(MATCHING_HEADER + "\n")
        for s1 in s1_ids:
            f.write(f"{s1}\t{','.join(by_s1.get(s1, []))}\n")
    return Path(path)


def verify_output_file(path: Path, expected_s1_ids: List[str], header: str, check_one_owner: bool = False) -> dict:
    """Read a written file back and check schema, one row per S1, no duplicate IDs in a list, ID prefixes."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    assert lines[0] == header, f"{path}: bad header {lines[0]!r}, expected {header!r}"
    rows = [ln.split("\t") for ln in lines[1:]]
    assert all(len(r) == 2 for r in rows), f"{path}: rows must have exactly 2 tab-separated columns"
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)), f"{path}: duplicate S1 rows"
    assert set(ids) == set(expected_s1_ids), f"{path}: S1 rows do not match the evaluated S1 set"
    n_ids = 0
    owners = []
    for s1, lst in rows:
        items = lst.split(",") if lst else []
        assert len(items) == len(set(items)), f"{path}: duplicate IDs in list for {s1}"
        assert all(i.startswith(("S2-", "S3-")) for i in items), f"{path}: non S2/S3 ID in list for {s1}"
        n_ids += len(items)
        owners += [(s1, i) for i in items]
    if check_one_owner:
        verify_one_owner(pd.DataFrame(owners, columns=["s1_id", "source_record_id"]))
    return {"rows": len(rows), "ids": n_ids}
