
from pathlib import Path

import joblib
import pandas as pd

from payment_platform.data.features import get_model_features
from payment_platform.data.generator import calculate_failure_probability


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"

MODELS = {
    "baseline": PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib",
    "candidate": PROJECT_ROOT / "models" / "logistic_regression_no_interactions.joblib",
}

PAYMENT_METHODS = [
    "upi",
    "credit_card",
    "debit_card",
    "net_banking",
    "wallet",
]

THRESHOLD = 0.05


def predict_routing(df: pd.DataFrame, model) -> pd.DataFrame:
    candidates = []

    for method in PAYMENT_METHODS:
        candidate = df.copy()
        candidate["payment_method"] = method
        candidate["candidate_method"] = method
        candidates.append(candidate)

    all_candidates = pd.concat(candidates, ignore_index=True)
    features = get_model_features(all_candidates)

    probabilities = model.predict_proba(features)[:, 1]
    all_candidates["predicted_success_probability"] = probabilities

    table = all_candidates.pivot(
        index="transaction_id",
        columns="candidate_method",
        values="predicted_success_probability",
    ).reset_index()

    current_methods = df[
        ["transaction_id", "payment_method"]
    ].rename(columns={"payment_method": "current_method"})

    table = table.merge(
        current_methods,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    table["current_success_probability"] = table.apply(
        lambda row: row[row["current_method"]],
        axis=1,
    )

    alternatives = table[PAYMENT_METHODS].copy()

    for method in PAYMENT_METHODS:
        alternatives[method] = alternatives[method].where(
            table["current_method"] != method
        )

    table["best_alternative_method"] = alternatives.idxmax(axis=1)
    table["best_alternative_success_probability"] = alternatives.max(axis=1)

    table["potential_improvement"] = (
        table["best_alternative_success_probability"]
        - table["current_success_probability"]
    )

    table["recommended"] = table["potential_improvement"] >= THRESHOLD

    return table[
        [
            "transaction_id",
            "best_alternative_method",
            "potential_improvement",
            "recommended",
        ]
    ]


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    
    # Calculate simulator ground-truth failure risk for each method.
    ground_truth_frames = []

    for method in PAYMENT_METHODS:
        candidates = df.copy()
        candidates["payment_method"] = method
        candidates["candidate_method"] = method
        ground_truth_frames.append(candidates)

    all_ground_truth = pd.concat(
        ground_truth_frames,
        ignore_index=True,
    )

    all_ground_truth["ground_truth_failure_probability"] = (
        calculate_failure_probability(all_ground_truth)
    )

    ground_truth_table = all_ground_truth.pivot(
        index="transaction_id",
        columns="candidate_method",
        values="ground_truth_failure_probability",
    ).reset_index()

    ground_truth_table = ground_truth_table.merge(
        df[["transaction_id", "payment_method"]].rename(
            columns={"payment_method": "current_method"}
        ),
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    ground_truth_table["current_failure_risk"] = (
        ground_truth_table.apply(
            lambda row: row[row["current_method"]],
            axis=1,
        )
    )

    print(f"Test transactions: {len(df):,}")
    print(f"Routing threshold: {THRESHOLD:.0%}")

    results = {}

    for name, model_path in MODELS.items():
        model = joblib.load(model_path)
        results[name] = predict_routing(df, model)
        print(f"Evaluated {name} model.")

    comparison = results["baseline"].merge(
        results["candidate"],
        on="transaction_id",
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )

    changed_method = (
        comparison["best_alternative_method_baseline"]
        != comparison["best_alternative_method_candidate"]
    )

    changed_recommendation = (
        comparison["recommended_baseline"]
        != comparison["recommended_candidate"]
    )

    baseline_count = comparison["recommended_baseline"].sum()
    candidate_count = comparison["recommended_candidate"].sum()

    # Evaluate both routing policies across the entire test population.
    ground_truth_table["oracle_alternative_failure_risk"] = (
        ground_truth_table.apply(
            lambda row: min(
                row[method]
                for method in PAYMENT_METHODS
                if method != row["current_method"]
            ),
            axis=1,
        )
    )

    
    comparison = comparison.merge(
        ground_truth_table[
            [
                "transaction_id",
                "current_failure_risk",
                "oracle_alternative_failure_risk",
                *PAYMENT_METHODS,
            ]
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )


    for policy in ["baseline", "candidate"]:
        chosen_methods = comparison[
            f"best_alternative_method_{policy}"
        ]

        
        comparison[f"{policy}_selected_failure_risk"] = comparison.apply(
            lambda row: row[row[f"best_alternative_method_{policy}"]],
            axis=1,
        )


        policy_risk = comparison["current_failure_risk"].where(
            ~comparison[f"recommended_{policy}"],
            comparison[f"{policy}_selected_failure_risk"],
        )

        comparison[f"{policy}_expected_failure_rate_after_routing"] = (
            policy_risk.mean()
        )

    baseline_risk = comparison["current_failure_risk"].mean()


    # Paired bootstrap confidence interval for the candidate's
    # improvement over baseline on the same test transactions.
    baseline_post_routing = comparison["current_failure_risk"].where(
        ~comparison["recommended_baseline"],
        comparison["baseline_selected_failure_risk"],
    )

    candidate_post_routing = comparison["current_failure_risk"].where(
        ~comparison["recommended_candidate"],
        comparison["candidate_selected_failure_risk"],
    )

    paired_improvement = (
        baseline_post_routing - candidate_post_routing
    ).to_numpy()

    import numpy as np

    rng = np.random.default_rng(42)
    n_bootstrap = 5000
    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(
            paired_improvement,
            size=len(paired_improvement),
            replace=True,
        )
        bootstrap_means[i] = sample.mean()

    ci_lower, ci_upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    print("\nPaired Bootstrap: Candidate vs Baseline")
    print("-" * 45)
    print(
        f"Mean expected risk improvement: "
        f"{paired_improvement.mean():.4%}"
    )
    print(
        f"95% bootstrap confidence interval: "
        f"[{ci_lower:.4%}, {ci_upper:.4%}]"
    )


    # Evaluate whether each model's recommendations reduce true
    # simulator failure risk by the configured threshold.
    for policy in ["baseline", "candidate"]:
        actual_risk_improvement = (
            comparison["current_failure_risk"]
            - comparison[f"{policy}_selected_failure_risk"]
        )

        recommended = comparison[f"recommended_{policy}"]

        genuinely_better = (
            recommended & (actual_risk_improvement > 0)
        )
        meets_threshold = (
            recommended & (actual_risk_improvement >= THRESHOLD)
        )

        recommendation_count = int(recommended.sum())

        print(f"\n{policy.title()} Recommendation Quality")
        print("-" * 40)
        print(f"Recommendations: {recommendation_count:,}")

        if recommendation_count:
            print(
                "Recommendations with lower true failure risk: "
                f"{genuinely_better.sum():,} "
                f"({genuinely_better.sum() / recommendation_count:.2%})"
            )
            print(
                "Recommendations achieving at least 5 pp risk reduction: "
                f"{meets_threshold.sum():,} "
                f"({meets_threshold.sum() / recommendation_count:.2%})"
            )
            print(
                "Mean true risk reduction among recommendations: "
                f"{actual_risk_improvement[recommended].mean():.4%}"
            )
    
    
    oracle_improvement = (
        comparison["current_failure_risk"]
        - comparison["oracle_alternative_failure_risk"]
    )
    oracle_recommended = oracle_improvement >= THRESHOLD

    oracle_risk_after_routing = comparison["current_failure_risk"].where(
        ~oracle_recommended,
        comparison["oracle_alternative_failure_risk"],
    ).mean()


    print("\nSynthetic Ground-Truth Routing Evaluation")
    print("=" * 55)
    print(f"Baseline expected failure rate: {baseline_risk:.4%}")
    
    print(
        f"Oracle expected failure rate after routing: "
        f"{oracle_risk_after_routing:.4%}"
    )
    print(
        f"Oracle expected risk reduction: "
        f"{baseline_risk - oracle_risk_after_routing:.4%}"
    )
    print(
        f"Oracle recommendations: {oracle_recommended.sum():,} "
        f"({oracle_recommended.mean():.2%})"
    )

    for policy in ["baseline", "candidate"]:
        after_routing = comparison["current_failure_risk"].where(
            ~comparison[f"recommended_{policy}"],
            comparison[f"{policy}_selected_failure_risk"],
        ).mean()

        print(
            f"{policy.title()} expected failure rate after routing: "
            f"{after_routing:.4%}"
        )
        print(
            f"{policy.title()} expected risk reduction: "
            f"{baseline_risk - after_routing:.4%}"
        )

    print("\nRouting Decision Comparison")
    print("=" * 55)
    print(f"Alternative method changed: {changed_method.sum():,} "
          f"({changed_method.mean():.2%})")
    print(f"Recommendation status changed: {changed_recommendation.sum():,} "
          f"({changed_recommendation.mean():.2%})")
    print(f"Baseline recommendations: {baseline_count:,} "
          f"({baseline_count / len(comparison):.2%})")
    print(f"Candidate recommendations: {candidate_count:,} "
          f"({candidate_count / len(comparison):.2%})")
    print(
        "Mean predicted improvement (baseline): "
        f"{comparison['potential_improvement_baseline'].mean():.4f}"
    )
    print(
        "Mean predicted improvement (candidate): "
        f"{comparison['potential_improvement_candidate'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
