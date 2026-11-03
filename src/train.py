from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[1]


def main():
    data_path = ROOT / "data/iris.csv"

    # Zapis danych Iris przy pierwszym uruchomieniu.
    iris = load_iris(as_frame=True)
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        iris.frame.to_csv(data_path, index=False)

    df = pd.read_csv(data_path)
    print("Pierwsze wiersze:\n", df.head())
    print("\nRozmiar:", df.shape)
    print("\nStatystyki:\n", df.describe())



if __name__ == "__main__":
    main()
