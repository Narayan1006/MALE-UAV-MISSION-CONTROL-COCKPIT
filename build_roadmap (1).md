# AeroTwin Aircraft Engine Digital Twin — Detailed Build Roadmap

This is the full step-by-step build plan, expanded from the base-level order already agreed on. No code here — this is the *what* and *how*, phase by phase, with acceptance criteria for each so you know when to move to the next one.

## What you already have (don't rebuild these)

| Asset | File | Status |
|---|---|---|
| Physics-informed engine simulator | `engine_simulator.py` | Built, tested, stable |
| Dataset column reference | `simulation_data_dictionary.md` | Complete |
| PS requirements breakdown | `_explained.md` | Complete |
| System architecture (ML live / LLM explain) | diagram from earlier | Agreed |

---

## Phase 1 — Project Setup (Day 0, ~1 hour)

**Folder structure:**
```
aerotwin-engine-digital-twin/
├── simulator/          # engine_simulator.py goes here
├── data/
│   ├── raw/             # generated mission CSVs
│   └── processed/       # windowed/feature-engineered versions
├── ml/                  # training scripts, saved models
├── backend/             # FastAPI app
├── dashboard/           # frontend
├── docs/                # data dictionary, PS explained, architecture, synopsis
├── requirements.txt
└── README.md
```

**Tasks:**
- `git init`, push to GitHub/GitLab, add all teammates as collaborators
- `requirements.txt`: numpy, pandas, scikit-learn, xgboost, torch, fastapi, uvicorn, matplotlib
- Move `engine_simulator.py` into `simulator/`, confirm it still runs and writes to `data/raw/`
- Write a one-paragraph README explaining the project (copy from _explained.md summary)

**Acceptance criteria:** Fresh clone of the repo + `pip install -r requirements.txt` + `python simulator/engine_simulator.py` produces CSVs with no errors.

**Owner:** Whoever is most comfortable with git — do this once, not per-person.

---

## Phase 2 — Full Dataset Generation (Day 1, ~2-3 hours)

The 9 CSVs already generated are a proof of concept, not enough volume to train on. You need variety.

**What to generate:**
- All 4 mission types × multiple random seeds (e.g., 10 seeds each) = 40 healthy runs
- Each of the 6 fault types × varied onset/failure timing × multiple seeds = aim for 15-20 runs per fault type
- Target: roughly 150-200 total mission CSVs, several hundred thousand labelled rows combined

**How:** Write a small loop script (not the simulator itself) that calls `run_mission()` from `engine_simulator.py` with varying `seed`, `fault_type`, `onset_s`, `failure_s` values, saving each to `data/raw/`.

**Split before touching ML:** Randomly assign mission CSVs (not individual rows — whole missions, so no data leakage) into `train/` (70%), `val/` (15%), `test/` (15%) subfolders.

**Acceptance criteria:** `data/raw/train`, `val`, `test` folders exist, each fault type is represented in all three splits, no single mission file appears in more than one split.

**Owner:** One person — this is a scripting task, not a team task.

---

## Phase 3 — Data Validation / Sanity Check (Day 1-2, ~2 hours)

Do not skip this. A bug caught here costs an hour; the same bug caught after training a model costs a day.

**Checks to run:**
- Plot `true_cht`, `egt`, `rpm`, `health_index` over time for 2-3 healthy missions and 2-3 faulted missions (matplotlib, side by side)
- Confirm: faulted sections visibly deviate from healthy baseline in the plots
- Confirm: no NaNs, no negative oil pressures, no CHT/EGT values wildly outside the ranges documented in `simulation_data_dictionary.md`
- Confirm: `fault_severity` actually ramps 0→1 over the configured window in faulted files, and stays 0 in healthy files
- Confirm: class balance — roughly how many rows are healthy vs each fault type (expect imbalance, that's normal and expected; note it for Phase 5)

**Acceptance criteria:** A short markdown or notebook with the plots + a written note confirming each check above passed (or what was fixed).

**Owner:** Same person as Phase 2, or whoever will do ML — they need to trust the data before modeling it.

---

## Phase 4 — Baseline Anomaly Detector (Day 2-3, ~1 day)

Do not start with LSTM. Start with the simplest thing that could possibly work.

**Approach:**
1. Compute rolling-window features per sensor channel (mean, std over a 30-60 second window) for each row
2. Train an Isolation Forest (or even a simple z-score threshold) **only on healthy mission data**
3. Score every row in the validation set — healthy rows should score "normal," faulted rows should score "anomalous," and the anomaly score should rise as `fault_severity` rises

**Evaluation:**
- Does the anomaly score correlate with `fault_severity`? (plot one against the other)
- Precision/recall treating `fault_type != "none"` as the positive class

**Decision point:** If this baseline already separates healthy from faulted reasonably well, you have a working anomaly detector — you can stop here for the demo and only upgrade to LSTM/autoencoder if time permits and you want a stronger story for judges. Don't build the complex version until the simple one is proven.

**Acceptance criteria:** A saved baseline model + a plot showing anomaly score rising with fault severity + a precision/recall number, however rough.

**Owner:** ML person #1.

---

## Phase 5 — Fault Classification + RUL Regression (Day 3-4, ~1-1.5 days)

**Fault classifier:**
- Features: same rolling-window stats from Phase 4
- Target: `fault_type` column (multi-class: none/injector/lubrication/cooling/misfire/sensor_drift/vibration)
- Model: XGBoost classifier — fast to train, handles imbalance reasonably with `scale_pos_weight` or class weighting
- Metric: per-class F1, plus a confusion matrix (some faults will be easier to distinguish than others — that's expected and worth mentioning honestly in documentation)

**RUL regressor:**
- Train only on rows where a fault is active (rul_seconds is only meaningful there)
- Target: `rul_seconds`, consider capping/normalizing (e.g., don't ask the model to predict beyond some max horizon — same trick used in CMAPSS-based research)
- Model: XGBoost regressor or a simple feedforward network
- Metric: MAE, RMSE — report these honestly, don't cherry-pick

**Acceptance criteria:** Both models saved, evaluated on the held-out test split (never touched during training), metrics written down for the documentation/synopsis.

**Owner:** ML person #2, working in parallel with Phase 4's owner.

---

## Phase 6 — API Layer (Day 4-5, ~1 day)

This is the single integration point — dashboard and LLM layer both talk to this, not to the models directly.

**Endpoint design (conceptually):**
- `POST /telemetry` — accepts one row (or small batch) of sensor readings, returns: anomaly score, predicted fault type + confidence, RUL estimate, health index
- `POST /mission/replay` — accepts a mission CSV, returns the full time-series of predictions for playback
- Internally: load the Phase 4 and Phase 5 models once at startup, not per-request

**Why this matters:** once this exists, the simulator becomes swappable for real CAN/ECU data later without touching the dashboard or ML code — exactly the "swappable data source" principle from the architecture.

**Acceptance criteria:** `curl` or Postman request to `/telemetry` with a sample row returns a valid JSON response with all four fields.

**Owner:** Backend person.

---

## Phase 7 — Minimal Dashboard (Day 5-6, ~1-1.5 days)

**Build in this order, not all at once:**
1. A page that calls `/mission/replay` with a stored CSV and plots the telemetry over time (line charts — RPM, CHT, EGT, health index)
2. Overlay the predicted anomaly/fault markers on the timeline
3. A simple alert panel — when fault confidence crosses a threshold, show it
4. Only after 1-3 work: polish the visuals, add mission-phase labels, styling

**Acceptance criteria:** Loading a faulted mission CSV in the dashboard visibly shows the fault being flagged at roughly the right point in the timeline.

**Owner:** Frontend person, can start on static UI in parallel with Phase 6 using dummy data, then swap in the real API once it's ready.

---

## Phase 8 — LLM Explain Layer (Day 6-7, ~half day)

Add this **last**, once Phases 4-7 are solid — it's the lowest-risk part and the easiest to cut if you're short on time.

**Approach:**
- When `/telemetry` or `/mission/replay` returns a flagged fault, pass fault type + confidence + RUL + recent sensor trend into a prompt
- Ask for a short, plain-language advisory (2-3 sentences: what's likely wrong, how confident, what to do)
- Keep this call outside any latency-critical path — it's for the dashboard's advisory panel, not for the live anomaly alert itself

**Acceptance criteria:** A flagged fault in the dashboard shows a readable, sensible advisory sentence, not just raw numbers.

**Owner:** Whoever's free — this is a small, contained task.

---

## Phase 9 — Polish + Stretch Goals (remaining time)

Only attempt these once Phases 1-8 all work end-to-end. Priority order if time is short:

1. **RUL uncertainty band** (show a range, not just a point estimate) — cheap, directly answers the "reliability engineering" technical expectation
2. **Explainability** (SHAP values on the XGBoost classifier) — cheap, directly answers "Explainable AI" desired innovation area
3. **Edge/onboard split** — split the anomaly detector (Phase 4) into a "lightweight onboard" version vs the full pipeline running on the ground station
4. **CAN bus interface** (`python-can` + virtual CAN) — swap the simulator's direct API calls for a CAN-message layer in between, to demonstrate the real deployment path
5. **Fleet view** — 3-4 simulated UAVs shown together on the dashboard
6. Documentation pass — technical roadmap doc, honest disclaimers section (reuse language from `_explained.md`), synopsis, PPT

---

## Suggested team split (5-6 people)

| Role | Phases owned |
|---|---|
| Simulator/data owner (you) | 1, 2, 3 |
| ML person #1 | 4 |
| ML person #2 | 5 |
| Backend person | 6 |
| Frontend person | 7 |
| Floating/support | 8, then whichever Phase 9 items get prioritized |

## Milestone checklist

- [ ] Phase 1 — repo runs clean on a fresh clone
- [ ] Phase 2 — 150-200 mission CSVs generated, split into train/val/test
- [ ] Phase 3 — data validated with plots, no bugs found (or fixed)
- [ ] Phase 4 — baseline anomaly detector working, score correlates with severity
- [ ] Phase 5 — fault classifier + RUL regressor evaluated on test set
- [ ] Phase 6 — API returns valid predictions for sample input
- [ ] Phase 7 — dashboard shows telemetry + flagged faults from real API
- [ ] Phase 8 — LLM advisory text shows up for flagged faults
- [ ] Phase 9 — pick 2-3 stretch items based on remaining time, don't try all of them

---

*This roadmap assumes a hackathon-scale timeline (roughly one week of focused work across a team). Adjust day estimates to your actual deadline — the phase order and dependencies matter more than the exact day counts.*
