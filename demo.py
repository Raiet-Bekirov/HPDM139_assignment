from dfair.core import differential_fairness


def demo() -> None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: pip install ucimlrepo"
        ) from exc

    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)

    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # metadata
    print(heart_disease.metadata)

    # variable information
    print(heart_disease.variables)

    # Example usage: compute DF on target labels by a sensitive attribute if present.
    # Replace 'sex' with the column you want to evaluate if needed.
    if "sex" in X.columns:
        outcomes = y.iloc[:, 0].tolist()
        groups = X["sex"].tolist()
        epsilon, _ = differential_fairness(outcomes, groups, alpha=1.0)
        print(f"\nDifferential Fairness epsilon: {epsilon:.4f}")
    else:
        print("\nNo 'sex' column found in features to compute fairness.")


if __name__ == "__main__":
    demo()
