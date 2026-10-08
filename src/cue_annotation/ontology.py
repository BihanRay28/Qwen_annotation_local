from .util import digest

ONTOLOGY_VERSION = "1.0"
CUES = {
    "LOOK_LECTURER_TEACHING_AREA": "Visible head/gaze orientation toward an identifiable teaching area.",
    "LOOK_PEER": "Visible orientation toward an identifiable nearby person; no inference about intent.",
    "LOOK_OWN_MATERIAL": "Visible orientation toward the target's identifiable book, paper or task surface.",
    "LOOK_OTHER_UNKNOWN": "An observable orientation whose target cannot be assigned to a named category; not hidden gaze.",
    "WRITE_OR_DO_ASSIGNMENT": "Visible writing or manipulation of identifiable task material; posture alone is insufficient.",
    "READ_OR_VIEW_MATERIAL": "Visible viewing of identifiable material; no comprehension claim.",
    "PHONE_USE": "An identifiable phone being visibly handled or viewed; academic relevance is not inferred.",
    "HEAD_DOWN": "Visible downward head posture, compatible with writing or reading.",
    "EYES_CLOSED": "Eye closure directly visible at sufficient resolution; hidden eyes are unknown.",
    "HAND_RAISED": "A visibly raised hand or arm, without inferred intention.",
    "APPARENT_PEER_CONVERSATION": "Temporally supported visible interaction; speech is not confirmed.",
    "PEER_COLLABORATION": "Visible joint actions with shared material/task; proximity alone is insufficient.",
    "OUT_OF_SEAT": "Visible evidence of being away from the seat; upright posture alone is insufficient.",
    "UNRELATED_OBJECT_ACTIVITY": "Visible object activity separate from identifiable current material; unresolved relevance is unknown.",
    "HEAD_OR_FACE_NOT_VISIBLE": "Head or face hidden/cropped out; other body-supported cues may remain assessable.",
    "STUDENT_NOT_OBSERVABLE": "The identified target cannot be visually observed; a missing crop file alone is not evidence.",
}
REQUIRES = {
    "LOOK_LECTURER_TEACHING_AREA": ("head",),
    "LOOK_PEER": ("head",),
    "LOOK_OWN_MATERIAL": ("head", "body"),
    "LOOK_OTHER_UNKNOWN": ("head",),
    "WRITE_OR_DO_ASSIGNMENT": ("hands",),
    "READ_OR_VIEW_MATERIAL": ("head",),
    "PHONE_USE": ("hands",),
    "HEAD_DOWN": ("head",),
    "EYES_CLOSED": ("eyes",),
    "HAND_RAISED": ("body",),
    "APPARENT_PEER_CONVERSATION": ("body",),
    "PEER_COLLABORATION": ("body",),
    "OUT_OF_SEAT": ("body",),
    "UNRELATED_OBJECT_ACTIVITY": ("hands",),
    "HEAD_OR_FACE_NOT_VISIBLE": ("body",),
    "STUDENT_NOT_OBSERVABLE": (),
}
RELATIONAL = frozenset(
    {
        "LOOK_LECTURER_TEACHING_AREA",
        "LOOK_PEER",
        "PEER_COLLABORATION",
        "APPARENT_PEER_CONVERSATION",
        "OUT_OF_SEAT",
        "UNRELATED_OBJECT_ACTIVITY",
    }
)
ONTOLOGY_FINGERPRINT = digest(
    {"version": ONTOLOGY_VERSION, "cues": CUES, "requires": REQUIRES, "relational": sorted(RELATIONAL)}
)
