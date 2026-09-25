from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/processed/transactions_v1.parquet")


def main() -> None:
    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)

    numeric_columns = [
        "amount",
        "retry_count",
        "transaction_velocity",
        "user_method_success_rate",
        "merchant_method_success_rate",
    ]

    print("\n=== NUMERIC DISTRIBUTIONS ===")
    summary = df[numeric_columns].describe().T
    summary["skew"] = df[numeric_columns].skew()
    print(summary.round(4).to_string())

    print("\n=== FAILURE RATE BY RETRY COUNT ===")
    retry_failure = (
        1 - df.groupby("retry_count")["payment_success"].mean()
    )
    retry_counts = df["retry_count"].value_counts().sort_index()

    retry_summary = pd.DataFrame(
        {
            "transactions": retry_counts,
            "failure_rate": retry_failure,
        }
    )
    retry_summary["failure_rate"] *= 100
    print(retry_summary.round(2).to_string())

    print("\n=== FAILURE RATE BY TRANSACTION VELOCITY ===")
    velocity_summary = (
        df.groupby("transaction_velocity")
        .agg(
            transactions=("payment_success", "size"),
            failure_rate=("payment_success", lambda x: (1 - x.mean()) * 100),
        )
    )
    print(velocity_summary.round(2).to_string())

    print("\n=== AMOUNT QUANTILE BINS ===")
    df["amount_bin"] = pd.qcut(
        df["amount"],
        q=10,
        duplicates="drop",
    )

    amount_summary = (
        df.groupby("amount_bin", observed=True)
        .agg(
            transactions=("payment_success", "size"),
            mean_amount=("amount", "mean"),
            failure_rate=("payment_success", lambda x: (1 - x.mean()) * 100),
        )
    )

    print(amount_summary.round(2).to_string())

    print("\n=== USER METHOD SUCCESS RATE BINS ===")
    df["user_rate_bin"] = pd.qcut(
        df["user_method_success_rate"],
        q=10,
        duplicates="drop",
    )

    user_rate_summary = (
        df.groupby("user_rate_bin", observed=True)
        .agg(
            transactions=("payment_success", "size"),
            mean_success_rate=("user_method_success_rate", "mean"),
            failure_rate=("payment_success", lambda x: (1 - x.mean()) * 100),
        )
    )

    print(user_rate_summary.round(4).to_string())

    print("\n=== MERCHANT METHOD SUCCESS RATE BINS ===")
    df["merchant_rate_bin"] = pd.qcut(
        df["merchant_method_success_rate"],
        q=10,
        duplicates="drop",
    )

    merchant_rate_summary = (
        df.groupby("merchant_rate_bin", observed=True)
        .agg(
            transactions=("payment_success", "size"),
            mean_success_rate=("merchant_method_success_rate", "mean"),
            failure_rate=("payment_success", lambda x: (1 - x.mean()) * 100),
        )
    )

    print(merchant_rate_summary.round(4).to_string())

    print("\n=== NUMERIC CORRELATION WITH PAYMENT SUCCESS ===")
    correlations = (
        df[numeric_columns + ["payment_success"]]
        .corr(numeric_only=True)["payment_success"]
        .drop("payment_success")
        .sort_values()
    )

    print(correlations.round(4).to_string())

    print("\n=== GROUND-TRUTH PROBABILITY VS ACTUAL OUTCOME ===")
    print(
        df[
            ["failure_probability", "payment_success"]
        ].corr(numeric_only=True).round(4)
    )

    print("\nNumeric EDA complete.")


if __name__ == "__main__":
    main()
    