from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]


def main():
    data_path = ROOT / "data/iris.csv"
    test_size = 0.2
    seed = 42

    if not 0 < test_size < 1:
        raise ValueError("TEST_SIZE musi być pomiędzy 0 i 1.")

    # Zapis danych Iris przy pierwszym uruchomieniu.
    iris = load_iris(as_frame=True)
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        iris.frame.to_csv(data_path, index=False)

    df = pd.read_csv(data_path)
    print("Pierwsze wiersze:\n", df.head())
    print("\nRozmiar:", df.shape)
    print("\nStatystyki:\n", df.describe())

    X = df.drop(columns="target")
    y = df["target"]
    # Stratyfikacja zachowuje proporcje klas.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    model = DecisionTreeClassifier(random_state=seed)
    model.fit(X_train, y_train)

    # Ocena na danych niewykorzystanych podczas treningu.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, labels=[0, 1, 2], target_names=iris.target_names,
        digits=4, zero_division=0
    )
    results = (
        f"Trening: {len(X_train)} próbek; test: {len(X_test)} próbek\n"
        f"TEST_SIZE={test_size}, RANDOM_SEED={seed}\n"
        f"Accuracy: {accuracy:.4f}\n\n{report}"
    )
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "wyniki.txt").write_text(results, encoding="utf-8")
    print("\n" + results)


if __name__ == "__main__":
    main()
