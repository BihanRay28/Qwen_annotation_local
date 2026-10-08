from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .ontology import CUES
from .schema import key, validate_record


def evaluate(annotations: Path, reference: Path) -> dict:
    humans = {}
    with reference.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("adjudicated") is not True:
                raise ValueError("Reference records must be independently adjudicated")
            if set(record["cues"]) != set(CUES) or any(
                v not in {"present", "not_observed", "unknown"} for v in record["cues"].values()
            ):
                raise ValueError("Every reference cue needs an explicit valid state")
            identity = key(record)
            if identity in humans:
                raise ValueError("Duplicate human reference key")
            humans[identity] = record
    if not humans:
        raise ValueError("Reference is empty")
    counts = {c: Counter() for c in CUES}
    matched = set()
    with annotations.open(encoding="utf-8") as stream:
        for line in stream:
            prediction = validate_record(json.loads(line)).model_dump()
            identity = key(prediction)
            if identity not in humans:
                continue
            if identity in matched:
                raise ValueError("Duplicate prediction for reference key")
            matched.add(identity)
            for cue in CUES:
                human, machine = humans[identity]["cues"][cue], prediction["cues"][cue]
                count = counts[cue]
                count["reviewed"] += 1
                if human == "unknown":
                    count["human_unknown"] += 1
                    count["claim_on_unknown"] += machine != "unknown"
                    continue
                count["eligible"] += 1
                count["answered"] += machine != "unknown"
                count["human_positive"] += human == "present"
                if human == "present":
                    count["tp" if machine == "present" else "fn"] += 1
                elif machine == "present":
                    count["fp"] += 1
                elif machine == "not_observed":
                    count["tn"] += 1
                else:
                    count["abstained_negative"] += 1
    if matched != set(humans):
        raise ValueError("Some human reference positions have no prediction; evaluation scope is incomplete")
    result = {}
    for cue, count in counts.items():
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        result[cue] = {
            **count,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "answer_coverage": count["answered"] / count["eligible"] if count["eligible"] else None,
            "claim_on_unknown_rate": count["claim_on_unknown"] / count["human_unknown"]
            if count["human_unknown"]
            else None,
        }
    return {
        "reviewed_positions": len(matched),
        "per_cue": result,
        "confidence_intervals": "not_computed; use session/track-window clustering in the frozen evaluation protocol",
    }
