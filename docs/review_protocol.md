# Pilot review and comparison

Stage 1 runs one complete track-window and checks its 250 expected positions, alignment, visibility, localisation, cue output, token and GPU resource measurements. Exercise graceful interruption and resume. Stage 2 selects five diverse track-windows with occlusion, ambiguous orientation, writing/reading and small-object activity if available. Stage 3 expands to approximately twenty classroom windows with selected tracks from retained sessions. Numeric first-N selection is convenient for engineering checks but does not guarantee diversity.

Independent raters inspect source images and assign cue states without seeing model predictions first. Review target identity, eligibility, mixed or fragmented tracks, evidence correctness, visibility, unknown handling and short events. Adjudicate disagreements. Fix material errors before freezing cue definitions, prompts, software, model and evaluation settings. Select cue-specific quality thresholds on development cases before assessing held-out cases or authorising bulk processing.

Create a template from a completed pilot:

```bash
docker run --rm --network none --entrypoint python \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$HOME/Desktop/eascca-cue-runs,dst=/outputs" \
  qwen-annotation-local:spark /workspace/scripts/create_approval_template.py \
  --run-dir /outputs/pilot-one --output /outputs/pilot-approval.json
```

The template is **unapproved**. After reviewing the broader pilot's quality and measured resource feasibility, an operator records `quality_reviewed=true`, `resources_feasible=true`, a reviewer and substantive notes. The bulk command verifies matching dataset, model artifact, configuration, ontology, prompt and schema fingerprints. A changed protocol invalidates this approval. The template itself is not evidence of human review, and automated success cannot substitute for this decision.

The proposed 3B-versus-7B comparison uses approximately 100–200 reviewed track-windows with identical imagery, cue definitions and a fixed protocol. Create a separate 3B configuration by changing `model_id` to `Qwen/Qwen2.5-VL-3B-Instruct`, download its weights and use a new run directory and its own pilot approval. Keep prompt-development cases separate from evaluation cases by session where possible. Otherwise retain complete sequences and potentially fragmented tracks of one person in the same partition, and report remaining dependence. Do not tune prompts repeatedly on the held-out cases.

Evaluation reference JSONL contains identifiers, all sixteen explicit cue states, and `adjudicated: true`. Example shape (fill the full cue mapping):

```text
{"session":5,"window":4,"track":2,"source_frame":750,"adjudicated":true,"cues":{...}}
```

Run `eascca-cues evaluate --annotations <frames.jsonl> --reference <adjudicated.jsonl> --output <metrics.json>`. The command reports cue-level precision, recall, F1, prevalence, denominators, answer coverage and claims made on human-unknown evidence. Model abstention on a human-visible positive counts as a missed positive. Human-unknown cases are excluded from binary detection metrics and inspected separately. Undefined metrics remain null, not invented zeros. Reference positions missing predictions cause an error rather than silent exclusion.

The evaluation helper does not compute confidence intervals or declare statistical significance. A frozen research analysis must account for session or track-window clustering and report rare-cue limits; millions of adjacent frames are not independent observations. Engineering completion, cue quality and educational interpretation are separate outcomes. A complete JSON run that repeatedly confuses people or overstates hidden behaviour fails the annotation objective.

