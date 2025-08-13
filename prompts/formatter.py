# prompts/formatter.py
# utilities for safe schema formatting and questionnaire rendering

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from schema import load_question_texts


def questions_block(qtexts: Dict[str, str]) -> str:
    """render ordered 'Qid: text' lines."""
    return "\n".join(f"{qid}: {text}" for qid, text in qtexts.items())


def mk_qids_placeholders(qids: Iterable[str]) -> Dict[str, str]:
    """generate placeholders for schema_block substitution."""
    scores = ", ".join([f'"{qid}": <int 1-9>' for qid in qids])
    evidence = ", ".join([f'"{qid}": "<non-empty string>"' for qid in qids])
    return {"qids_scores": scores, "qids_evidence": evidence}


def safe_schema_format(template: str, mapping: Dict[str, str]) -> str:
    """escape json braces except placeholders, then format."""
    protected = (("{qids_scores}", "«QS»"), ("{qids_evidence}", "«QE»"))
    tmp = template
    for needle, token in protected:
        tmp = tmp.replace(needle, token)

    tmp = tmp.replace("{", "{{").replace("}", "}}")

    for needle, token in protected:
        tmp = tmp.replace(token, needle)

    return tmp.format(**mapping)


def require_keys(obj: Dict, keys: Iterable[str], where: str) -> None:
    for k in keys:
        if k not in obj:
            raise KeyError(f"missing key '{k}' in {where}")


def load_qtexts(qpath: str | Path) -> Dict[str, str]:
    qtexts = load_question_texts(Path(qpath))
    if not qtexts:
        raise ValueError("questionnaire is empty or failed to load")
    return qtexts
