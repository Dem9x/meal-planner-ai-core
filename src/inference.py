import joblib
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"

vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
df = joblib.load(MODEL_DIR / "meals_database.pkl")


def filter_meals(
    data,
    allergies=None,
    diet=None,
    max_cooking_time=None,
    budget=None,
    meal_type=None,
    cuisine=None
):
    filtered = data.copy()
    allergies = allergies or []

    for allergy in allergies:
        allergy = allergy.strip().lower()
        if allergy:
            filtered = filtered[
                ~filtered["allergen_tags"].str.lower().str.contains(allergy, na=False)
            ]

    if diet:
        filtered = filtered[
            filtered["diet_tags"].str.lower().str.contains(diet.lower(), na=False)
        ]

    if max_cooking_time:
        filtered = filtered[
            filtered["cooking_time"] <= max_cooking_time
        ]

    if budget:
        filtered = filtered[
            filtered["budget_level"].str.lower() == budget.lower()
        ]

    if meal_type:
        filtered = filtered[
            filtered["meal_type"].str.lower() == meal_type.lower()
        ]

    if cuisine:
        cuisine_filtered = filtered[
            filtered["cuisine"].str.lower().str.contains(cuisine.lower(), na=False)
        ]

        if len(cuisine_filtered) >= 1:
            filtered = cuisine_filtered

    return filtered


def recommend_meals(user_profile, top_n=5):
    query = " ".join([
        str(user_profile.get("meal_type", "")),
        str(user_profile.get("diet", "")),
        str(user_profile.get("budget", "")),
        str(user_profile.get("cuisine", "")),
        str(user_profile.get("goal", "")),
        str(user_profile.get("difficulty", ""))
    ])

    filtered_df = filter_meals(
        df,
        allergies=user_profile.get("allergies", []),
        diet=user_profile.get("diet"),
        max_cooking_time=user_profile.get("max_cooking_time"),
        budget=user_profile.get("budget"),
        meal_type=user_profile.get("meal_type"),
        cuisine=user_profile.get("cuisine")
    )

    if filtered_df.empty:
        return []

    filtered_vectors = vectorizer.transform(filtered_df["features"])
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, filtered_vectors).flatten()

    result = filtered_df.copy()
    result["score"] = scores

    result = result.sort_values("score", ascending=False).head(top_n)

    return result[
        [
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
            "score"
        ]
    ].to_dict("records")


if __name__ == "__main__":
    user_profile = {
        "meal_type": "dinner",
        "diet": "halal",
        "budget": "low",
        "cuisine": "indonesian",
        "goal": "high_protein balanced",
        "difficulty": "easy",
        "allergies": ["peanut"],
        "max_cooking_time": 30
    }

    results = recommend_meals(user_profile, top_n=5)

    print("Top recommendations:")
    for meal in results:
        print(
            f"- {meal['name']} | "
            f"{meal['calories']} kcal | "
            f"{meal['protein']}g protein | "
            f"{meal['cooking_time']} min | "
            f"score: {round(meal['score'], 4)}"
        )