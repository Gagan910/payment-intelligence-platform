# Payment Intelligence & Smart Routing Platform

An ML-powered payment decisioning platform that estimates payment-failure risk and evaluates whether switching payment methods is likely to improve the expected outcome. This portfolio project uses **synthetic payment data only**.

## Live Demo

- **Frontend checkout:** https://payment-intelligence-frontend.onrender.com
- **Backend API / Swagger:** https://payment-intelligence-platform.onrender.com/docs
- **Health check:** https://payment-intelligence-platform.onrender.com/health

> **Safety:** This is a simulation. It does not process real money. Never enter real card numbers, CVVs, UPI PINs, OTPs, bank credentials, or other payment secrets.

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

### Experiment Interpretation

The experiment ID is `payment_routing_v1`. The summary endpoint reports control/treatment counts, attempted and successful transactions, unresolved transactions, success rates, confidence intervals, and estimated lift.

During manual verification, the live experiment was marked `insufficient_data` because both variants were below the configured minimum sample size of 100. Its observed lift is **not evidence of a validated real-world improvement**.

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
| `GET` | `/experiments/{experiment_id}/summary` | Retrieve experiment summary metrics |

## Technology Stack

- **Language:** Python 3.12
- **API:** FastAPI
- **Machine learning:** scikit-learn / Logistic Regression
- **Data processing:** pandas, NumPy
- **Database:** PostgreSQL
- **Project configuration:** `pyproject.toml`
- **Testing:** automated test suite
- **Deployment/containerization:** Docker and Render
- **Frontend:** deployed checkout interface

Confirm exact dependency versions and the application entry point from the repository configuration before publishing version-specific setup instructions.

## Run Locally

These instructions provide a starting point. Database settings, required environment variables, dependency groups, and the ASGI entry point must match the current repository configuration.

### 1. Clone the repository

```bash
git clone https://github.com/Gagan910/payment-intelligence-platform.git
cd payment-intelligence-platform
```

Replace the placeholder with your actual repository URL.

### 2. Create a virtual environment

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

### 3. Install the project

Inspect `pyproject.toml` for the declared build backend and development dependencies. If editable installation is supported, a typical command is:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Install the project's test/development dependency group if one is defined.

### 4. Configure the database and environment

Set the required database connection and application settings using the environment-variable names defined by the project. Never commit `.env` files, passwords, API keys, or production credentials.

# Initialize the database schema (PostgreSQL must be running)
psql "$env:DATABASE_URL" -f db/schema.sql

# Apply the payment-attempt uniqueness migration
psql "$env:DATABASE_URL" -f db/migrations/001_add_payment_attempt_uniqueness.sql

### 5. Start the API

Start the FastAPI backend from the project root:

```powershell
uvicorn payment_platform.api.app:app --reload

```bash
uvicorn payment_platform.api.app:app --reload
```

Verify that module path before running the command. If the server uses the default port, Swagger UI is available at `http://127.0.0.1:8000/docs`.

## Testing

The recorded development verification included **109 passing tests** and successful Python compilation of the experiment scripts. Re-run the tests in the current checkout before relying on this count for a release.

A common test command is:

```bash
pytest
```

Configure the project's test database and required environment variables first. Do not point tests at production data.

## Deployed Demo Walkthrough

1. Open the [frontend checkout](https://payment-intelligence-frontend.onrender.com).
2. Use only synthetic values.
3. Use [Swagger UI](https://payment-intelligence-platform.onrender.com/docs) to create a synthetic transaction.
4. Copy the returned transaction ID and use it with `POST /predict`.
5. Use the same ID with `POST /recommend`.
6. If applicable, record the result with `POST /recommend/decision`.
7. Record a simulated attempt with `POST /payment-attempts`.
8. Query `/experiments/payment_routing_v1/summary` to inspect aggregate experiment metrics.

## Limitations

- Data and outcomes are synthetic and depend on generator assumptions.
- Model performance on synthetic data may not generalize to real payment systems.
- Experiment results need adequate sample sizes; insufficient data cannot establish a reliable lift.
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
- Add structured audit logging and further production controls.

## Author

**Gagan Sharma**

- GitHub: https://github.com/Gagan910
- Portfolio: https://Gagan910.github.io

---

*For educational and portfolio demonstration purposes; uses synthetic payment data.*
