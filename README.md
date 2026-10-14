# Payment Intelligence & Smart Routing Platform

An ML-powered payment decisioning platform that estimates payment-failure risk and evaluates whether switching payment methods is likely to improve the expected outcome. This portfolio project uses **synthetic payment data only**.

## Live Demo

- **Frontend checkout:** https://payment-intelligence-frontend.onrender.com
- **Backend API / Swagger:** https://payment-intelligence-platform.onrender.com/docs
- **Health check:** https://payment-intelligence-platform.onrender.com/health

> **Safety:** This is a simulation and does not process real money. Never enter real card numbers, CVVs, UPI PINs, OTPs, bank credentials, or other payment secrets.

## Problem Statement

Payment attempts may fail because of payment method, network conditions, transaction context, or historical success patterns. This project demonstrates how machine learning and a policy-based routing layer can work together to:

1. Estimate a transaction's payment-failure probability.
2. Compare the current payment method with alternatives.
3. Recommend switching only when the predicted improvement meets a configurable threshold.
4. Record recommendations and simulated payment attempts.
5. Summarize experiment outcomes and identify insufficient sample sizes.

The routing layer does **not** blindly choose the method with the highest predicted success probability. The deployed example uses a minimum-improvement threshold of **5 percentage points**.

## Key Features

- Synthetic transaction-data generation.
- Binary payment success/failure prediction with probability outputs.
- Logistic Regression baseline and model/feature comparison experiments.
- Payment-method recommendation using a configurable improvement threshold.
- APIs for transactions, predictions, recommendations, decisions, payment attempts, and experiment summaries.
- Temporal evaluation, calibration analysis, routing backtests, feature ablation, and robustness experiments.
- A/B experiment assignment and evaluation with confidence intervals and significance testing.
- Automated tests, Docker support, and deployed frontend/backend.

## Architecture

```text
Synthetic transaction context
          |
          v
    Checkout frontend
          |
          v
       FastAPI API
          |
          +--> Persist transaction
          |
          +--> Predict failure probability
          |          |
          |          v
          |    Logistic Regression model
          |
          +--> Compare payment methods
          |          |
          |          v
          |    Routing policy / threshold
          |
          +--> Record recommendation decision
          |
          +--> Record simulated payment attempt
          |
          v
     PostgreSQL persistence
          |
          v
     Experiment summary
```

This is a conceptual overview. Use the live Swagger documentation to inspect the current API contract.

## Supported Payment Methods

- UPI
- Credit card
- Debit card
- Net banking
- Wallet

All payment activity is simulated. The application does not connect to payment rails.

## Machine Learning

### Prediction

The model estimates payment-failure probability from transaction context. The prediction API returns failure and success probabilities, plus a model version.

Example response from a synthetic demonstration transaction:

```json
{
  "transaction_id": "txn_demo_1001",
  "failure_probability": 0.1241,
  "success_probability": 0.8759,
  "model_version": "logistic_regression_baseline_v1"
}
```

This is an example response, **not a general performance claim**.

### Evaluation

The project uses time-based train/validation/test separation to reduce temporal leakage. Evaluation scripts cover:

- ROC-AUC and PR-AUC
- Precision, recall, and F1
- Brier score and calibration analysis
- Confusion matrix
- Feature ablation and model comparisons
- Routing threshold backtests and robustness checks

Results must be interpreted in the context of the synthetic data generator. Performance on synthetic data does not establish performance on real payment traffic.

### Routing Policy

The recommendation layer evaluates whether an alternative offers enough predicted improvement over the current method. The deployed configuration uses a minimum required improvement of **5 percentage points**. If no alternative meets the threshold, the API can return `KEEP_CURRENT`.

## A/B Experimentation

The experiment ID is `payment_routing_v1`. The framework supports control/treatment assignment and evaluation of payment outcomes, routing switches, confidence intervals, and estimated lift.

### Preliminary local synthetic experiment

Run ID: `ab_20261014_212801_7ac7e38a`

| Metric | Control | Treatment |
|---|---:|---:|
| Transactions | 49 | 51 |
| Successful payments | 28 | 37 |
| Payment success rate | 57.14% | 72.55% |
| 95% confidence interval for success rate | 43.27%–69.98% | 59.05%–82.89% |
| Routing switches | 0 | 25 |
| Mean payment latency | 34.738 ms | 35.159 ms |

**Observed absolute difference:** +15.41 percentage points
**95% confidence interval for the difference:** −3.09 to +33.90 percentage points
**Two-proportion z-test p-value:** 0.1064
**Statistically significant at α = 0.05:** No

The treatment group had a higher observed success rate in this 100-transaction run, but the confidence interval includes zero and the p-value exceeds 0.05. This experiment therefore does **not** establish a statistically significant improvement.

**Interpretation limitations:**

- This was a small synthetic experiment; its results are not evidence of real-world payment performance.
- The experiment runner marks treatment switches as accepted, so the reported 100% acceptance rate does not represent natural user behavior.
- Abandonment rate is unavailable because abandonment events are not recorded.
- Time-to-success is not currently defined as an experiment metric.
- This local experiment is separate from the deployed experiment summary. During prior manual verification, the deployed experiment reported `insufficient_data` because both variants were below the configured minimum sample size of 100.

## API Endpoints

The deployed Swagger UI is the authoritative source for request schemas and response codes:

**https://payment-intelligence-platform.onrender.com/docs**

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `HEAD` | `/health` | Health check without response body |
| `POST` | `/transactions` | Create a synthetic transaction |
| `POST` | `/predict` | Predict risk for a persisted transaction |
| `POST` | `/recommend` | Evaluate routing options and return a recommendation |
| `POST` | `/recommend/decision` | Record whether a recommendation was accepted |
| `POST` | `/payment-attempts` | Record a simulated payment attempt |
| `GET` | `/experiments/{experiment_id}/summary` | Retrieve aggregate experiment metrics |

## Technology Stack

- **Language:** Python 3.12
- **API:** FastAPI and Uvicorn
- **Machine learning:** scikit-learn, Logistic Regression, XGBoost
- **Data processing:** pandas and NumPy
- **Database:** PostgreSQL with psycopg
- **Configuration:** `pyproject.toml`, environment variables, and python-dotenv
- **Testing:** pytest and HTTPX-compatible test tooling declared in the project configuration
- **Deployment/containerization:** Docker and Render
- **Frontend:** Deployed checkout interface

The declared dependency versions are pinned in `pyproject.toml`.

## Run Locally

These instructions assume Python 3.12 and a running PostgreSQL instance. Use a dedicated local development/test database; do not point development tests or synthetic experiment runs at production data.

### 1. Clone the repository

```bash
git clone https://github.com/Gagan910/payment-intelligence-platform.git
cd payment-intelligence-platform
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install the project and test dependencies

```bash
python -m pip install --upgrade pip
pip install -e ".[test]"
```

### 4. Configure the database

Set `DATABASE_URL` to a PostgreSQL database dedicated to this local instance. The application loads environment variables from `.env` when present. The fallback URL in the source is:

```text
postgresql://postgres:postgres@127.0.0.1:5432/payment_intelligence
```

Create a local `.env` file with your own database URL, for example:

```dotenv
DATABASE_URL=postgresql://<user>:<password>@127.0.0.1:5432/<database_name>
```

Replace the placeholders with your local database settings. Do not commit `.env` files, passwords, API keys, or production credentials.

Initialize the schema and apply the migration:

```powershell
psql "$env:DATABASE_URL" -f db/schema.sql
psql "$env:DATABASE_URL" -f db/migrations/001_add_payment_attempt_uniqueness.sql
```

Run these commands only against the intended local database. Confirm the migration's expected starting schema before applying it to any existing database.

### 5. Start the API

From the repository root, with the virtual environment active:

```powershell
uvicorn payment_platform.api.app:app --reload
```

On macOS/Linux, use the same command in the activated environment. With the default Uvicorn port, Swagger UI is available at `http://127.0.0.1:8000/docs`.

### 6. Run tests

Configure `DATABASE_URL` to point to a dedicated test database before running tests:

```bash
python -m pytest -q
```

A recorded development verification completed with **109 passing tests** against a local Docker PostgreSQL test database. This is a historical result; re-run the suite on the current checkout to verify it independently.

## Deployed Demo Walkthrough

1. Open the [frontend checkout](https://payment-intelligence-frontend.onrender.com).
2. Use only synthetic values.
3. Use [Swagger UI](https://payment-intelligence-platform.onrender.com/docs) to create a synthetic transaction.
4. Copy the returned transaction ID and use it with `POST /predict`.
5. Use the same ID with `POST /recommend`.
6. If applicable, record the decision with `POST /recommend/decision`.
7. Record a simulated attempt with `POST /payment-attempts`.
8. Query `/experiments/payment_routing_v1/summary` to inspect aggregate experiment metrics.

## Limitations

- Data and outcomes are synthetic and depend on generator assumptions.
- Model performance on synthetic data may not generalize to real payment systems.
- The observed A/B difference was not statistically significant in the preliminary 100-transaction run.
- The experiment runner forces switch acceptance; natural user acceptance is not measured.
- Abandonment and time-to-success metrics are not currently available.
- Experiment results need adequate sample sizes; insufficient data cannot establish reliable lift.
- This is a portfolio simulation, not a production payment processor.
- No real payments, bank integrations, card processing, or payment-network authorization are performed.

## Security

- Never submit real card numbers, CVVs, UPI PINs, OTPs, bank passwords, or API secrets.
- Keep credentials in environment variables or a secret manager.
- Use synthetic identifiers and test values for demonstrations.
- Review CORS, authentication, rate limiting, database access, and logs before any use beyond a controlled demo.

## Potential Future Improvements

- Add a transaction-history view and operational monitoring.
- Monitor prediction drift and probability calibration.
- Evaluate on approved, anonymized real-world data if legally permitted.
- Collect sufficient experiment samples before drawing conclusions.
- Record natural recommendation acceptance and abandonment events.
- Add structured audit logging and further production controls.

## Author

**Gagan Sharma**

- GitHub: https://github.com/Gagan910
- Portfolio: https://Gagan910.github.io

---

*For educational and portfolio demonstration purposes; uses synthetic payment data.*
