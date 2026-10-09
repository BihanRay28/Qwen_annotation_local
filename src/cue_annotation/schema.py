from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .ontology import CUES, RELATIONAL, REQUIRES
from .util import digest

SCHEMA_VERSION = "1.0"
VisibilityState = Literal["visible", "partial", "not_visible", "unknown"]
CueState = Literal["present", "not_observed", "unknown"]


class PlaceholderEvidenceError(ValueError):
    """A known instruction/example was copied instead of describing image evidence."""


def model_json_text(text: str) -> str:
    # Accept only one complete optional JSON fence. Never extract/repair partial JSON,
    # strip surrounding prose, change frame IDs, or invent missing records.
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", stripped, re.DOTALL | re.IGNORECASE)
    return fenced[1].strip() if fenced else stripped


def check_evidence_descriptions(evidence):
    for item in evidence:
        normalised = " ".join(item.description.casefold().split()).rstrip(".! ")
        if normalised == "concrete visible evidence for this target":
            raise PlaceholderEvidenceError(
                "Model copied example/placeholder evidence instead of describing the supplied images"
            )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Visibility(StrictModel):
    body: VisibilityState
    head: VisibilityState
    face: VisibilityState
    eyes: VisibilityState
    hands: VisibilityState


class Evidence(StrictModel):
    frames: list[StrictInt] = Field(min_length=1)
    description: str = Field(min_length=1, max_length=1500)


class ModelFrame(StrictModel):
    frame: StrictInt
    visibility: Visibility
    present: list[str]
    unknown: list[str]
    confidence: Literal["low", "medium", "high"]
    evidence: list[Evidence]
    review_flags: list[str] = Field(default_factory=list)


class ModelResponse(StrictModel):
    frames: list[ModelFrame]


class Localisation(StrictModel):
    status: Literal["accepted", "rejected", "unavailable"]
    bbox: list[StrictInt] | None = None
    score: float | None = None
    margin: float | None = None
    reason: str | None = None


class FrameRecord(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    session: StrictInt
    window: StrictInt
    track: StrictInt
    source_frame: StrictInt
    time_seconds: float
    input_status: Literal["matched", "crop_missing", "global_missing", "both_missing"]
    global_path: str | None
    crop_path: str | None
    global_sha256: str | None
    crop_sha256: str | None
    localisation: Localisation
    visibility: Visibility
    cues: dict[str, CueState]
    confidence: Literal["low", "medium", "high"]
    evidence: list[Evidence]
    review_flags: list[str]
    track_eligibility: Literal["unreviewed"] = "unreviewed"


def key(record: FrameRecord | dict) -> str:
    value = record.model_dump() if isinstance(record, FrameRecord) else record
    return f"{value['session']}:{value['window']}:{value['track']}:{value['source_frame']}"


def validate_record(value: dict) -> FrameRecord:
    record = FrameRecord.model_validate(value)
    check_evidence_descriptions(record.evidence)
    if set(record.cues) != set(CUES):
        raise ValueError("A complete explicit cue-state mapping is required")
    if min(record.session, record.window) < 1 or min(record.track, record.source_frame) < 0:
        raise ValueError("Invalid source identifiers")
    if abs(record.time_seconds - record.source_frame / 25) > 1e-9:
        raise ValueError("Invalid nominal source time")
    for modality in ("global", "crop"):
        path = getattr(record, f"{modality}_path")
        sha = getattr(record, f"{modality}_sha256")
        if (path is None) != (sha is None) or (
            sha is not None and (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha))
        ):
            raise ValueError("Source path and SHA256 must be paired")
    actual_status = (
        "matched"
        if record.global_path and record.crop_path
        else "crop_missing"
        if record.global_path
        else "global_missing"
        if record.crop_path
        else "both_missing"
    )
    if actual_status != record.input_status:
        raise ValueError("Input status contradicts source availability")
    loc = record.localisation
    if loc.status == "accepted" and (
        record.input_status != "matched"
        or loc.bbox is None
        or len(loc.bbox) != 4
        or min(loc.bbox) < 0
        or loc.bbox[2] <= loc.bbox[0]
        or loc.bbox[3] <= loc.bbox[1]
        or loc.score is None
        or loc.margin is None
    ):
        raise ValueError("Accepted localisation requires matched inputs and valid native coordinates/scores")
    if record.input_status in {"crop_missing", "both_missing"} and any(
        s != "unknown" for s in record.cues.values()
    ):
        raise ValueError("Unidentified target must not receive cue labels")
    for cue, state in record.cues.items():
        if state != "unknown" and not assessable(cue, record.visibility, record.localisation):
            raise ValueError("Cue state violates visibility/localisation eligibility")
    if any(s == "present" for s in record.cues.values()) and not any(
        record.source_frame in e.frames for e in record.evidence
    ):
        raise ValueError("Present cue lacks current-frame evidence")
    return record


def assessable(cue: str, visibility: Visibility, loc: Localisation) -> bool:
    if cue in RELATIONAL and loc.status != "accepted":
        return False
    if cue == "HEAD_OR_FACE_NOT_VISIBLE":
        return (
            visibility.body in {"visible", "partial"}
            and visibility.head != "unknown"
            and visibility.face != "unknown"
        )
    if cue == "STUDENT_NOT_OBSERVABLE":
        return visibility.body != "unknown" and loc.status == "accepted"
    return all(getattr(visibility, part) == "visible" for part in REQUIRES[cue])


def base_record(source: dict, loc: dict) -> dict:
    return {
        "session": source["session"],
        "window": source["window"],
        "track": source["track"],
        "source_frame": source["source_frame"],
        "time_seconds": source["time_seconds"],
        "input_status": source["input_status"],
        "localisation": loc,
        **{
            f"{kind}_{field}": (
                source[f"{kind}_source"]["path" if field == "path" else "sha256"]
                if source[f"{kind}_source"]
                else None
            )
            for kind in ("global", "crop")
            for field in ("path", "sha256")
        },
    }


def missing_record(source: dict) -> dict:
    value = base_record(source, {"status": "unavailable", "reason": "target_crop_missing"})
    value.update(
        visibility={part: "unknown" for part in Visibility.model_fields},
        cues={c: "unknown" for c in CUES},
        confidence="low",
        evidence=[],
        review_flags=["source_missing", "target_identity_unresolved"],
    )
    return validate_record(value).model_dump()


def expand_response(
    text: str, sources: list[dict], allowed_frames: set[int], locations: dict[int, dict]
) -> list[dict]:
    response = ModelResponse.model_validate_json(model_json_text(text))
    expected = {s["source_frame"] for s in sources}
    frames = [f.frame for f in response.frames]
    if len(frames) != len(expected) or set(frames) != expected:
        raise ValueError("Model output must own each requested primary position exactly once")
    by_frame = {f.frame: f for f in response.frames}
    result = []
    for source in sources:
        frame = by_frame[source["source_frame"]]
        check_evidence_descriptions(frame.evidence)
        present, unknown = set(frame.present), set(frame.unknown)
        if len(present) != len(frame.present) or len(unknown) != len(frame.unknown):
            raise ValueError("Repeated cue code")
        if present & unknown or (present | unknown) - set(CUES):
            raise ValueError("Unknown or conflicting cue codes")
        for evidence in frame.evidence:
            if not evidence.description.strip() or not set(evidence.frames) <= allowed_frames:
                raise ValueError("Evidence cites an unavailable or out-of-chunk frame")
        if present and not any(frame.frame in e.frames for e in frame.evidence):
            raise ValueError("Present cues need evidence citing the primary frame itself")
        loc = Localisation.model_validate(locations[frame.frame])
        states = {}
        flags = set(frame.review_flags) | {"eligibility_review_required"}
        for cue in CUES:
            state = "present" if cue in present else "unknown" if cue in unknown else "not_observed"
            if state != "unknown" and not assessable(cue, frame.visibility, loc):
                state = "unknown"
                flags.add("visibility_or_localisation_gated")
            states[cue] = state
        if (
            states["HEAD_OR_FACE_NOT_VISIBLE"] == "present"
            and frame.visibility.head == frame.visibility.face == "visible"
        ):
            raise ValueError("Hidden head/face cue contradicts visibility")
        if states["STUDENT_NOT_OBSERVABLE"] == "present" and frame.visibility.body in {"visible", "partial"}:
            raise ValueError("Not-observable cue contradicts visible target")
        if loc.status != "accepted":
            flags.add("weak_localisation")
        elif loc.reason:
            flags.add(loc.reason)
        value = base_record(source, loc.model_dump())
        value.update(
            visibility=frame.visibility.model_dump(),
            cues=states,
            confidence=frame.confidence,
            evidence=[e.model_dump() for e in frame.evidence],
            review_flags=sorted(flags),
        )
        result.append(validate_record(value).model_dump())
    return result


SCHEMA_FINGERPRINT = digest(
    {
        "record": FrameRecord.model_json_schema(),
        "response": ModelResponse.model_json_schema(),
        "response_transport_version": "1.1",
        "evidence_validation_version": "1.1",
    }
)
