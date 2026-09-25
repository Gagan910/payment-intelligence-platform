from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASET_PATH = Path("data/processed/transactions_v1.parquet")
OUTPUT_DIR = Path("data/processed/eda")


def main() -> None:
    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Transaction amount distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["amount"], bins=50)
    plt.xlabel("Transaction amount")
    plt.ylabel("Number of transactions")
    plt.title("Transaction Amount Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amount_distribution.png", dpi=150)
    plt.close()

    # 2. Failure probability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["failure_probability"], bins=50)
    plt.xlabel("Ground-truth failure probability")
    plt.ylabel("Number of transactions")
    plt.title("Failure Probability Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_probability_distribution.png", dpi=150)
    plt.close()

    # 3. Payment method distribution
    payment_counts = df["payment_method"].value_counts()

    plt.figure(figsize=(10, 6))
    payment_counts.plot(kind="bar")
    plt.xlabel("Payment method")
    plt.ylabel("Number of transactions")
    plt.title("Payment Method Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "payment_method_distribution.png", dpi=150)
    plt.close()

    # 4. Failure rate by payment method
    failure_by_method = (
        1 - df.groupby("payment_method")["payment_success"].mean()
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    (failure_by_method * 100).plot(kind="bar")
    plt.xlabel("Payment method")
    plt.ylabel("Failure rate (%)")
    plt.title("Failure Rate by Payment Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_rate_by_payment_method.png", dpi=150)
    plt.close()

    # 5. Failure rate by network quality
    failure_by_network = (
        1 - df.groupby("network_quality")["payment_success"].mean()
    )

    network_order = [
        "poor",
        "average",
        "good",
        "excellent",
    ]

    failure_by_network = failure_by_network.reindex(network_order)

    plt.figure(figsize=(10, 6))
    (failure_by_network * 100).plot(kind="bar")
    plt.xlabel("Network quality")
    plt.ylabel("Failure rate (%)")
    plt.title("Failure Rate by Network Quality")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_rate_by_network.png", dpi=150)
    plt.close()

    # 6. Failure rate by hour
    failure_by_hour = (
        1 - df.groupby("hour_of_day")["payment_success"].mean()
    )

    plt.figure(figsize=(10, 6))
    (failure_by_hour * 100).plot(kind="line", marker="o")
    plt.xlabel("Hour of day")
    plt.ylabel("Failure rate (%)")
    plt.title("Failure Rate by Hour of Day")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_rate_by_hour.png", dpi=150)
    plt.close()

    print("\nEDA completed successfully.")
    print(f"Charts saved to: {OUTPUT_DIR}")

    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"- {file.name}")


if __name__ == "__main__":
    main()