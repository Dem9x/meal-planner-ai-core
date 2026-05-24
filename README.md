---
license: apache-2.0
tags:
  - meal-planning
  - recommender-system
  - nutrition
  - tfidf
  - cosine-similarity
  - scikit-learn
  - food-recommendation
language:
  - en
pipeline_tag: tabular-classification
library_name: scikit-learn
---

# Meal Planner AI Recommender

Meal Planner AI Recommender is a lightweight AI meal recommendation system that generates personalized meal suggestions based on user preferences such as diet, budget, allergies, cuisine, cooking time, and nutrition goals.

This model uses a content-based recommendation approach with TF-IDF vectorization and cosine similarity.

## Model Overview

This project is not a large language model. It is a lightweight recommendation engine designed to be used as the model core for a meal planner application.

The model recommends meals based on:

- Diet preference
- Allergy restrictions
- Budget level
- Cuisine preference
- Meal type
- Cooking time
- Nutrition goals
- Difficulty level

## Files

| File | Description |
|---|---|
| `vectorizer.pkl` | Trained TF-IDF vectorizer |
| `meals_database.pkl` | Processed meal database |
| `inference.py` | Example inference script |
| `week_plan.json` | Example generated 7-day meal plan |
| `shopping_list.json` | Example generated shopping list |
| `requirements.txt` | Python dependencies |

## Dataset Columns

The meal database contains these fields:

```txt
id
name
meal_type
calories
protein
carbs
fat
ingredients
diet_tags
allergen_tags
budget_level
cooking_time
cuisine
difficulty
features