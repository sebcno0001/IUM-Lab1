# Laboratorium 1 - Inżynieria uczenia maszynowego

Klasyfikacja zbioru Iris za pomocą drzewa decyzyjnego.

Repozytorium: https://github.com/sebcno0001/IUM-Lab1

## Uruchomienie

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

W Windows: `.venv\Scripts\activate`.

Utwórz `.env` na podstawie `.env.example` i ustaw:

```dotenv
DATA_PATH=data/iris.csv
TEST_SIZE=0.2
RANDOM_SEED=42
```

Puste wartości korzystają z powyższych ustawień domyślnych.

```bash
python src/train.py
```

Skrypt tworzy CSV przy pierwszym uruchomieniu, wyświetla statystyki,
trenuje model i zapisuje ocenę w `results/wyniki.txt`.

## Struktura

- `src/` - kod źródłowy
- `data/` - lokalne dane CSV
- `results/` - wyniki oceny modelu
- `models/` - katalog na modele
- `notebooks/` - pusty katalog wymagany w strukturze
- `requirements.txt` - wersje zależności
- `.env.example` - szablon konfiguracji

Dane, modele, środowisko `.venv` i lokalny `.env` są wykluczone z Gita.
