from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Meal Planner AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"

vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
df = joblib.load(MODEL_DIR / "meals_database.pkl")
used_meals = set()


class UserProfile(BaseModel):
    diet: Optional[str] = "halal"
    budget: Optional[str] = "low"
    cuisine: Optional[str] = "indonesian"
    goal: Optional[str] = "balanced"
    difficulty: Optional[str] = "easy"
    allergies: List[str] = []
    max_cooking_time: Optional[int] = 30
    include_snack: Optional[bool] = True


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
        return pd.DataFrame()

    filtered_vectors = vectorizer.transform(filtered_df["features"])
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, filtered_vectors).flatten()

    result = filtered_df.copy()
    result["score"] = scores

    return result.sort_values("score", ascending=False).head(top_n)


def pick_meal(user_profile, meal_type, top_n=8):
    global used_meals

    profile = user_profile.copy()
    profile["meal_type"] = meal_type

    recommendations = recommend_meals(profile, top_n=top_n)

    if recommendations.empty:
        return None

    fresh_recommendations = recommendations[
        ~recommendations["name"].isin(used_meals)
    ]

    if fresh_recommendations.empty:
        fresh_recommendations = recommendations

    selected = fresh_recommendations.sample(1).iloc[0]
    used_meals.add(selected["name"])

    return {
        "id": str(selected["id"]),
        "name": str(selected["name"]),
        "meal_type": str(selected["meal_type"]),
        "calories": int(selected["calories"]),
        "protein": int(selected["protein"]),
        "carbs": int(selected["carbs"]),
        "fat": int(selected["fat"]),
        "ingredients": str(selected["ingredients"]),
        "diet_tags": str(selected["diet_tags"]),
        "allergen_tags": str(selected["allergen_tags"]),
        "budget_level": str(selected["budget_level"]),
        "cooking_time": int(selected["cooking_time"]),
        "cuisine": str(selected["cuisine"]),
        "difficulty": str(selected["difficulty"]),
        "score": round(float(selected["score"]), 4)
    }


def generate_day_plan(user_profile, include_snack=True):
    meal_types = ["breakfast", "lunch", "dinner"]

    if include_snack:
        meal_types.append("snack")

    day_plan = {
        "meals": {},
        "daily_total": {
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0
        }
    }

    for meal_type in meal_types:
        meal = pick_meal(user_profile, meal_type)
        day_plan["meals"][meal_type] = meal

        if meal:
            day_plan["daily_total"]["calories"] += meal["calories"]
            day_plan["daily_total"]["protein"] += meal["protein"]
            day_plan["daily_total"]["carbs"] += meal["carbs"]
            day_plan["daily_total"]["fat"] += meal["fat"]

    return day_plan


def generate_shopping_list(week_plan):
    ingredients = []

    for day in week_plan["days"]:
        for meal in day["meals"].values():
            if meal and meal.get("ingredients"):
                meal_ingredients = [
                    item.strip().lower()
                    for item in str(meal["ingredients"]).split(",")
                    if item.strip()
                ]
                ingredients.extend(meal_ingredients)

    shopping_counts = {}

    for item in ingredients:
        shopping_counts[item] = shopping_counts.get(item, 0) + 1

    return [
        {
            "ingredient": item,
            "frequency": count
        }
        for item, count in sorted(
            shopping_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ]


def generate_week_plan(user_profile, include_snack=True):
    global used_meals
    used_meals = set()

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    week_plan = {
        "user_profile": user_profile,
        "days": []
    }

    for day in days:
        day_plan = generate_day_plan(user_profile, include_snack=include_snack)
        day_plan["day"] = day
        week_plan["days"].append(day_plan)

    week_plan["shopping_list"] = generate_shopping_list(week_plan)

    return week_plan


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Meal Planner AI API is running"
    }


@app.post("/recommend")
def recommend(profile: UserProfile):
    user_profile = profile.dict()
    user_profile["meal_type"] = "dinner"

    recommendations = recommend_meals(user_profile, top_n=5)

    if recommendations.empty:
        return {
            "results": []
        }

    return {
        "results": recommendations.to_dict("records")
    }


@app.post("/generate-meal-plan")
def generate_meal_plan(profile: UserProfile):
    user_profile = profile.dict()
    include_snack = user_profile.pop("include_snack", True)

    result = generate_week_plan(
        user_profile=user_profile,
        include_snack=include_snack
    )

    return result

@app.post("/generate-meal-plan")
def generate_meal_plan(profile: UserProfile):
    user_profile = profile.dict()
    include_snack = user_profile.pop("include_snack", True)

    return generate_week_plan(
        user_profile=user_profile,
        include_snack=include_snack
    )