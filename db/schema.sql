-- Payment Intelligence & Smart Routing Platform
-- PostgreSQL database schema

CREATE TABLE users (
    user_id VARCHAR(100) PRIMARY KEY,
    user_segment VARCHAR(50) NOT NULL,
    preferred_payment_method VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE merchants (
    merchant_id VARCHAR(100) PRIMARY KEY,
    merchant_category VARCHAR(100) NOT NULL,
    merchant_risk_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE transactions (
    transaction_id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL REFERENCES users(user_id),
    merchant_id VARCHAR(100) NOT NULL REFERENCES merchants(merchant_id),
    amount NUMERIC(12, 2) NOT NULL CHECK (amount > 0),
    currency VARCHAR(10) NOT NULL DEFAULT 'INR',
    timestamp TIMESTAMPTZ NOT NULL,
    selected_payment_method VARCHAR(50) NOT NULL,
    experiment_variant VARCHAR(50),
    status VARCHAR(50) NOT NULL
);

CREATE TABLE transaction_context (

    transaction_id VARCHAR(100) PRIMARY KEY
        REFERENCES transactions(transaction_id)
        ON DELETE CASCADE,

    device_type VARCHAR(50) NOT NULL,

    network_quality VARCHAR(50) NOT NULL,

    retry_count INTEGER NOT NULL
        CHECK (retry_count >= 0),

    transaction_velocity INTEGER NOT NULL
        CHECK (transaction_velocity >= 0),

    user_method_success_rate DOUBLE PRECISION NOT NULL
        CHECK (user_method_success_rate >= 0
            AND user_method_success_rate <= 1),

    merchant_method_success_rate DOUBLE PRECISION NOT NULL
        CHECK (merchant_method_success_rate >= 0
            AND merchant_method_success_rate <= 1)

);

CREATE TABLE predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) NOT NULL REFERENCES transactions(transaction_id),
    model_version VARCHAR(100) NOT NULL,
    failure_probability DOUBLE PRECISION NOT NULL
        CHECK (failure_probability >= 0 AND failure_probability <= 1),
    success_probability DOUBLE PRECISION NOT NULL
        CHECK (success_probability >= 0 AND success_probability <= 1),
    prediction_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    latency_ms DOUBLE PRECISION
);

CREATE TABLE recommendations (
    recommendation_id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) NOT NULL REFERENCES transactions(transaction_id),
    original_method VARCHAR(50) NOT NULL,
    recommended_method VARCHAR(50),
    recommendation_score DOUBLE PRECISION,
    reason TEXT,
    accepted BOOLEAN,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE payment_attempts (
    attempt_id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) NOT NULL REFERENCES transactions(transaction_id),
    payment_method VARCHAR(50) NOT NULL,
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    outcome VARCHAR(50) NOT NULL
);

CREATE TABLE experiment_assignments (
    assignment_id BIGSERIAL PRIMARY KEY,
    experiment_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    variant VARCHAR(50) NOT NULL,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE experiment_events (
    event_id BIGSERIAL PRIMARY KEY,
    experiment_id VARCHAR(100) NOT NULL,
    transaction_id VARCHAR(100),
    event_type VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Indexes for common transaction and experiment queries.

CREATE INDEX idx_transactions_user_id
    ON transactions(user_id);

CREATE INDEX idx_transactions_merchant_id
    ON transactions(merchant_id);

CREATE INDEX idx_transactions_timestamp
    ON transactions(timestamp);

CREATE INDEX idx_predictions_transaction_id
    ON predictions(transaction_id);

CREATE INDEX idx_recommendations_transaction_id
    ON recommendations(transaction_id);

CREATE INDEX idx_payment_attempts_transaction_id
    ON payment_attempts(transaction_id);

CREATE INDEX idx_experiment_events_experiment_id
    ON experiment_events(experiment_id);

CREATE INDEX idx_experiment_events_transaction_id
    ON experiment_events(transaction_id);