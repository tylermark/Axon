# REVIEW.md — Axon 2-Step Review System

## Overview

All code must pass **two sequential review stages** before merging to `main`. No exceptions. This mirrors the thesis model's own philosophy: just as the framework chains SAT constraints → topology checks → physics validation before accepting output, the code itself must survive architecture review → engineering review before acceptance.

```
Agent writes code
  → Self-check (agent verifies against its own scope in AGENTS.md)
  → QA Agent writes/runs tests
  → Step 1: Architecture Review (review-arch)
  → Step 2: Engineering Review (review-eng)
  → Merge to main
```

---

## Step 1: Architecture Review (`review-arch`)

**Reviewer:** Architecture Reviewer Agent  
**Trigger:** Agent opens a PR from `dev/<agent-id>/<feature>` → `main`  
**Branch rename:** PR branch moves to `review/<agent-id>/<feature>` during review  

### Review Checklist

The Architecture Reviewer evaluates the following. Every item must pass.

#### 1.1 — Mathematical Fidelity
- [ ] Does the implementation match the thesis model specification in `MODEL_SPEC.md`?
- [ ] Are mathematical formulations (loss functions, attention equations, diffusion schedules) implemented correctly?
- [ ] Are numerical stability concerns addressed (epsilon values, gradient clipping, log-sum-exp tricks)?
- [ ] Do variable names map clearly to the mathematical notation in the spec?

#### 1.2 — Interface Contract Compliance
- [ ] Does the module consume exactly the input type defined in `docs/interfaces/`?
- [ ] Does the module produce exactly the output type defined in `docs/interfaces/`?
- [ ] Are all fields populated — no `None` where the contract requires a value?
- [ ] Are tensor shapes documented and verified (assertions or runtime checks)?

#### 1.3 — Scope Boundaries
- [ ] Does the agent stay within its declared scope in `AGENTS.md`?
- [ ] Are there zero imports from modules outside the agent's scope (except interface types)?
- [ ] Does the code avoid duplicating logic that belongs to another agent?

#### 1.4 — Architectural Soundness
- [ ] Is the design extensible without modifying existing code (open/closed principle)?
- [ ] Are there clear separation points if a component needs to be swapped (e.g., different vision backbone)?
- [ ] Is differentiability maintained through the entire chain where required?
- [ ] Are there no gradient-breaking operations (`detach()`, `numpy()` calls mid-graph)?

#### 1.5 — Design Documentation
- [ ] Is there an ADR in `docs/decisions/` for any non-obvious design choice?
- [ ] Are complex algorithms documented with inline references to thesis sections or paper citations?

### Step 1 Outcomes

| Outcome | Action |
|---------|--------|
| **APPROVE** | Proceeds to Step 2 |
| **REQUEST CHANGES** | Returns to agent with specific, actionable feedback. Agent fixes and re-submits to Step 1. |
| **REJECT** | Fundamental design flaw. Agent must re-approach the problem. Requires ADR documenting why the approach was rejected and what the new direction is. |

### Step 1 Review Template

```markdown
## Architecture Review — [Module/Feature]

**Reviewer:** review-arch
**Agent:** [agent-id]
**Branch:** review/[agent-id]/[feature]
**Date:** YYYY-MM-DD

### Mathematical Fidelity
- Status: PASS / FAIL
- Notes:

### Interface Compliance
- Status: PASS / FAIL
- Notes:

### Scope Boundaries
- Status: PASS / FAIL
- Notes:

### Architectural Soundness
- Status: PASS / FAIL
- Notes:

### Design Documentation
- Status: PASS / FAIL
- Notes:

### Verdict: APPROVE / REQUEST CHANGES / REJECT
### Required Actions:
1.
2.
```

---

## Step 2: Engineering Review (`review-eng`)

**Reviewer:** Engineering Reviewer Agent  
**Trigger:** Step 1 passes (Architecture Reviewer approves)  
**Prerequisite:** QA Agent test suite must already pass for the relevant module  

### Review Checklist

The Engineering Reviewer evaluates the following. Every item must pass.

#### 2.1 — Code Quality
- [ ] Passes `ruff check` with zero warnings
- [ ] Passes `ruff format --check` (consistent formatting)
- [ ] All public functions have Google-style docstrings with type annotations
- [ ] No dead code, commented-out blocks, or TODO items without linked tasks
- [ ] Meaningful variable and function names — no single-letter names except in mathematical contexts (where they match notation)

#### 2.2 — Performance
- [ ] No unnecessary memory copies (`.clone()`, `.detach().cpu().numpy()` where avoidable)
- [ ] Batch operations preferred over loops where possible
- [ ] GPU-aware: tensors stay on device through computation chains
- [ ] Large allocations are profiled or annotated with expected sizes
- [ ] No O(n²) algorithms where O(n log n) alternatives exist for expected data sizes

#### 2.3 — Edge Cases & Error Handling
- [ ] Empty inputs handled gracefully (empty PDF, zero paths, degenerate graph)
- [ ] Numerical edge cases addressed: division by zero, NaN propagation, empty batches
- [ ] Meaningful error messages with context (not bare `raise ValueError`)
- [ ] Logging at appropriate levels (DEBUG for internals, INFO for pipeline stages, WARNING for recoverable issues, ERROR for failures)

#### 2.4 — Test Coverage
- [ ] QA Agent's tests cover all public API surfaces
- [ ] Edge cases from 2.3 have corresponding test cases
- [ ] Integration test exists for each interface boundary this module participates in
- [ ] Coverage meets or exceeds 90% for the module

#### 2.5 — Security & Reliability
- [ ] No hardcoded paths, credentials, or magic numbers without named constants
- [ ] File I/O uses context managers (`with` statements)
- [ ] External library calls are version-pinned in dependencies
- [ ] No unbounded memory growth (check for accumulating lists in loops)

#### 2.6 — Reproducibility
- [ ] Random seeds are configurable and passed explicitly (not global state)
- [ ] Non-deterministic operations are documented
- [ ] Configuration is externalized (not hardcoded)

### Step 2 Outcomes

| Outcome | Action |
|---------|--------|
| **APPROVE** | Code merges to `main`. |
| **REQUEST CHANGES** | Returns to agent for fixes. Does NOT require re-passing Step 1 unless changes are architectural. |
| **ESCALATE** | Engineering review surfaces an architectural concern missed in Step 1. Returns to Architecture Reviewer for re-evaluation. |

### Step 2 Review Template

```markdown
## Engineering Review — [Module/Feature]

**Reviewer:** review-eng
**Agent:** [agent-id]
**Branch:** review/[agent-id]/[feature]
**Date:** YYYY-MM-DD
**Step 1 Approved By:** review-arch on YYYY-MM-DD

### Code Quality
- Status: PASS / FAIL
- Notes:

### Performance
- Status: PASS / FAIL
- Notes:

### Edge Cases & Error Handling
- Status: PASS / FAIL
- Notes:

### Test Coverage
- Coverage: XX%
- Status: PASS / FAIL
- Notes:

### Security & Reliability
- Status: PASS / FAIL
- Notes:

### Reproducibility
- Status: PASS / FAIL
- Notes:

### Verdict: APPROVE / REQUEST CHANGES / ESCALATE
### Required Actions:
1.
2.
```

---

## Review Flow Diagram

```
                    ┌──────────────────┐
                    │  Agent writes    │
                    │  code on dev/    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Agent self-     │
                    │  check vs scope  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  QA Agent writes │
                    │  & runs tests    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  STEP 1          │
                    │  Architecture    │◄──── REQUEST CHANGES ────┐
                    │  Review          │                          │
                    └────────┬─────────┘                          │
                             │ APPROVE                            │
                    ┌────────▼─────────┐                          │
                    │  STEP 2          │                          │
                    │  Engineering     │──── REQUEST CHANGES ─────┘
                    │  Review          │       (non-architectural)
                    └────────┬─────────┘
                             │ APPROVE        ┌───────────────────┐
                             │                │  STEP 1 RE-EVAL   │
                             │   ESCALATE ───►│  (if arch issue   │
                             │                │   found in Step 2)│
                    ┌────────▼─────────┐      └───────────────────┘
                    │  MERGE to main   │
                    └──────────────────┘
```

---

## Special Review Rules

### Interface Changes
If any agent proposes a change to a contract in `docs/interfaces/`:
1. The change must be reviewed by **both** the producing and consuming agents
2. Architecture Reviewer has final approval
3. All downstream agents must update their code before the interface change merges

### Cross-Module PRs
If a task requires changes across multiple modules:
1. Integration Agent coordinates the PR
2. Each module's changes are reviewed by the relevant Builder Agent for correctness
3. Architecture Reviewer evaluates the cross-module interaction
4. Engineering Reviewer evaluates the combined changeset

### Hotfixes
For critical bugs blocking other agents:
1. Hotfix can skip Step 1 if the change is purely mechanical (typo, off-by-one, missing import)
2. Step 2 is still required
3. Hotfix must be retroactively reviewed in Step 1 within 24 hours
4. Hotfix branches: `hotfix/<agent-id>/<description>`

---

## Review Metrics

Track across the project lifetime:
- **First-pass approval rate** — target: >70% at Step 1, >85% at Step 2
- **Average review cycles** — target: <2 round-trips per PR
- **Escalation rate** — target: <5% of Step 2 reviews escalate to Step 1
- **Time in review** — monitor but no hard target (quality over speed)
