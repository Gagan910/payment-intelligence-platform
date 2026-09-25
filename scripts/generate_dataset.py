from pathlib import Path

from payment_platform.data.generator import (
    generate_transactions,
    simulate_payment_outcomes,
)


DEFAULT_ROWS = 100_000
DEFAULT_SEED = 42

OUTPUT_PATH = Path("data/processed/transactions_v1.parquet")


def main() -> None:
    print("Generating synthetic payment dataset...")

    transactions = generate_transactions(
        n_transactions=DEFAULT_ROWS,
        seed=DEFAULT_SEED,
    )

    transactions = simulate_payment_outcomes(
        transactions,
        seed=DEFAULT_SEED,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transactions.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nDataset generated successfully.")
    print(f"Rows: {len(transactions):,}")
    print(f"Columns: {len(transactions.columns)}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MB")

    print(
        f"Overall failure rate: "
        f"{1 - transactions['payment_success'].mean():.4f}"
    )


if __name__ == "__main__":
    main()