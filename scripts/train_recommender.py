from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "meals_300_indonesian.csv"
MODEL_DIR = ROOT_DIR / "model"

VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
MEALS_DATABASE_PATH = MODEL_DIR / "meals_database.pkl"


REQUIRED_COLUMNS = [
    "id",
    "name",
    "meal_type",
    "calories",
    "protein",
    "carbs",
    "fat",
    "ingredients",
    "diet_tags",
    "allergen_tags",
    "budget_level",
    "cooking_time",
    "cuisine",
    "difficulty",
]


NUMERIC_COLUMNS = [
    "calories",
    "protein",
    "carbs",
    "fat",
    "cooking_time",
]


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df = df.fillna("")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    text_columns = [
        "id",
        "name",
        "meal_type",
        "ingredients",
        "diet_tags",
        "allergen_tags",
        "budget_level",
        "cuisine",
        "difficulty",
    ]

    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    df["features"] = (
        df["meal_type"].astype(str) + " " +
        df["name"].astype(str) + " " +
        df["ingredients"].astype(str) + " " +
        df["diet_tags"].astype(str) + " " +
        df["allergen_tags"].astype(str) + " " +
        df["budget_level"].astype(str) + " " +
        df["cuisine"].astype(str) + " " +
        df["difficulty"].astype(str)
    )

    return df


def train_vectorizer(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )

    meal_vectors = vectorizer.fit_transform(df["features"])

    return vectorizer, meal_vectors


def save_model(vectorizer, df: pd.DataFrame) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(df, MEALS_DATABASE_PATH)


def print_report(df: pd.DataFrame, meal_vectors) -> None:
    print("=== Meal Planner Recommender Training Report ===")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Vector shape: {meal_vectors.shape}")
    print()

    print("Meal type distribution:")
    print(df["meal_type"].value_counts())
    print()

    print("Budget distribution:")
    print(df["budget_level"].value_counts())
    print()

    print("Cuisine distribution:")
    print(df["cuisine"].value_counts().head(20))
    print()

    print("Saved files:")
    print(f"- {VECTORIZER_PATH}")
    print(f"- {MEALS_DATABASE_PATH}")
    print()
    print("Training completed successfully.")


def main() -> None:
    df = load_dataset()
    df = clean_dataset(df)
    vectorizer, meal_vectors = train_vectorizer(df)
    save_model(vectorizer, df)
    print_report(df, meal_vectors)


if __name__ == "__main__":
    main()