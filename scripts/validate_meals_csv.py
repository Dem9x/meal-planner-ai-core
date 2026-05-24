import pandas as pd

PATH = "dataset/meals_300_indonesian.csv"

required_columns = [
    "id","name","meal_type","calories","protein","carbs","fat",
    "ingredients","diet_tags","allergen_tags","budget_level",
    "cooking_time","cuisine","difficulty"
]

valid_meal_types = {"breakfast", "lunch", "dinner", "snack"}
valid_budget = {"low", "medium", "high"}
valid_difficulty = {"easy", "medium"}
forbidden = ["pork", "bacon", "ham", "alcohol", "wine", "beer"]

allergen_rules = {
    "soy": ["tofu", "tempeh", "soy sauce"],
    "egg": ["egg"],
    "peanut": ["peanut", "peanut sauce"],
    "gluten": ["noodles", "noodle", "bread", "flour", "pasta"],
    "milk": ["milk", "cheese", "yogurt"],
    "fish": ["fish", "tuna", "sardine", "mackerel", "anchovy"],
    "shellfish": ["shrimp", "prawn"],
    "coconut": ["coconut milk", "coconut"],
    "sesame": ["sesame"],
}

df = pd.read_csv(PATH).fillna("")
errors = []
warnings = []

if len(df) != 300:
    errors.append(f"Expected 300 rows, got {len(df)}")

missing = [c for c in required_columns if c not in df.columns]
if missing:
    errors.append(f"Missing columns: {missing}")

if not missing:
    if df["id"].duplicated().any():
        errors.append("Duplicate IDs found")
    if df["name"].duplicated().any():
        duplicated = df[df["name"].duplicated()]["name"].tolist()
        errors.append(f"Duplicate meal names found: {duplicated[:10]}")

    bad_meal_types = set(df["meal_type"]) - valid_meal_types
    if bad_meal_types:
        errors.append(f"Invalid meal_type values: {bad_meal_types}")

    bad_budget = set(df["budget_level"]) - valid_budget
    if bad_budget:
        errors.append(f"Invalid budget_level values: {bad_budget}")

    bad_difficulty = set(df["difficulty"]) - valid_difficulty
    if bad_difficulty:
        errors.append(f"Invalid difficulty values: {bad_difficulty}")

    for col in ["calories", "protein", "carbs", "fat", "cooking_time"]:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            errors.append(f"Invalid numeric values in {col}")

    if (df["ingredients"].str.strip() == "").any():
        errors.append("Some rows have empty ingredients")

    if (df["diet_tags"].str.strip() == "").any():
        errors.append("Some rows have empty diet_tags")

    if (df["cuisine"].str.lower() == "indonesian").sum() < 230:
        errors.append("Cuisine indonesian must be at least 230 rows")

    if (df["budget_level"].str.lower() == "low").sum() < 170:
        errors.append("Budget low must be at least 170 rows")

    for word in forbidden:
        mask = df["ingredients"].str.lower().str.contains(word, na=False)
        if mask.any():
            errors.append(f"Forbidden ingredient found: {word}")

    for _, row in df.iterrows():
        ingredients = str(row["ingredients"]).lower()
        tags = str(row["allergen_tags"]).lower()
        for required_tag, keywords in allergen_rules.items():
            if any(k in ingredients for k in keywords) and required_tag not in tags:
                warnings.append(
                    f"{row['id']} {row['name']}: ingredients contain {keywords}, but allergen_tags missing {required_tag}"
                )

print("=== meals_300_indonesian.csv validation report ===")
print(f"Total rows: {len(df)}")
print("\nMeal type distribution:")
print(df["meal_type"].value_counts())

print("\nBudget distribution:")
print(df["budget_level"].value_counts())

print("\nCuisine distribution:")
print(df["cuisine"].value_counts().head(20))

print("\nErrors:")
if errors:
    for e in errors:
        print(f"- {e}")
else:
    print("- None")

print("\nWarnings:")
if warnings:
    for w in warnings[:50]:
        print(f"- {w}")
    if len(warnings) > 50:
        print(f"... and {len(warnings) - 50} more warnings")
else:
    print("- None")

if errors:
    raise SystemExit("Validation failed")

print("\nSUCCESS: meals_300_indonesian.csv is valid")
