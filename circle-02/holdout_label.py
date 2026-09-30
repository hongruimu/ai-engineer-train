"""Circle 2 独立测试集盲标入口。"""

from __future__ import annotations

import json
import pathlib
import sys

import label as blind_label


HERE = pathlib.Path(__file__).resolve().parent
HOLDOUT_FILE = HERE / "golden_holdout.json"
SESSION_FILE = HERE / ".holdout_label_session.json"
SEED = 20260930

CANDIDATES = {
    "T01": ["C2", "C1", "B2", "H1"],
    "T02": ["B1", "G1", "H2", "C1"],
    "T03": ["G2", "B2", "F3"],
    "T04": ["G3", "B1", "E3"],
    "T05": ["F3", "F4", "C2"],
    "T06": ["D1", "I1", "B1"],
    "T07": ["I1", "D1"],
    "T08": ["B2", "C2", "E3"],
    "T09": ["E2", "B2", "G1"],
    "T10": ["E1", "B1", "B2"],
    "T11": ["D2", "B2", "G1"],
    "T12": ["F4", "C1", "H1"],
    "T13": ["B1", "G1", "G2"],
    "T14": ["E3", "B2", "G2"],
    "T15": ["C1", "H1", "H2", "H3", "B1"],
}


def configure() -> None:
    blind_label.GOLDEN_FILE = HOLDOUT_FILE
    blind_label.SESSION_FILE = SESSION_FILE
    blind_label.SEED = SEED
    blind_label.CANDIDATES = CANDIDATES


def main() -> None:
    configure()
    docs = json.loads(blind_label.CORPUS_FILE.read_text("utf-8"))
    session = blind_label.load_session([doc["doc_id"] for doc in docs])
    blind_label.save_session(session)

    if len(sys.argv) > 1 and sys.argv[1] == "commit":
        blind_label.commit(session)
    elif len(sys.argv) > 1 and sys.argv[1] == "focus":
        blind_label.focus(session, sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        blind_label.show(session)


if __name__ == "__main__":
    main()
