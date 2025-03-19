import pandas as pd
import spacy
import random


def load_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)

    # Preprocess missing values
    df.dropna(subset=['recipe_name', 'ingredients', 'directions'], inplace=True)  # Remove incomplete recipes
    df.fillna("Unknown", inplace=True)  # Fill other missing values

    return df


def extract_ingredients(text, nlp):
    """Extract key ingredients using spaCy NLP."""
    doc = nlp(text)
    ingredients = [ent.text for ent in doc.ents if ent.label_ in ['FOOD', 'INGREDIENT']]
    return ingredients if ingredients else [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN']]


def generate_recipe(ingredients, df):
    """Find recipes based on one or multiple ingredients."""
    ingredients = [ingredient.lower() for ingredient in ingredients]
    filtered_recipes = df[df['ingredients'].str.contains('|'.join(ingredients), case=False, na=False)]

    if not filtered_recipes.empty:
        recipe = filtered_recipes.sample(1).iloc[0]
        return f"Recipe: {recipe['recipe_name']}\nIngredients: {recipe['ingredients']}\nInstructions: {recipe['directions']}"
    else:
        return suggest_similar_recipes(ingredients, df)


def suggest_similar_recipes(ingredients, df):
    """Suggest recipes with similar ingredients if an exact match isn't found."""
    matched_recipes = df[df['ingredients'].apply(lambda x: any(ing in x.lower() for ing in ingredients))]
    if not matched_recipes.empty:
        recipe = matched_recipes.sample(1).iloc[0]
        return f"No exact match found. Try this instead:\n\nRecipe: {recipe['recipe_name']}\nIngredients: {recipe['ingredients']}\nInstructions: {recipe['directions']}"
    else:
        return "No recipe found. Try a different ingredient."


def get_random_recipe(df):
    """Get a completely random recipe."""
    if df.empty:
        return "No recipes available."
    recipe = df.sample(1).iloc[0]
    return f"Recipe: {recipe['recipe_name']}\nIngredients: {recipe['ingredients']}\nInstructions: {recipe['directions']}"


def filter_by_diet(df, diet_type):
    """Filter recipes by dietary preference (e.g., Vegetarian, Vegan, Non-Vegetarian)."""
    if diet_type.lower() == 'vegetarian':
        return df[df['ingredients'].str.contains('chicken|fish|meat|beef|pork', case=False, na=False) == False]
    elif diet_type.lower() == 'vegan':
        return df[df['ingredients'].str.contains('chicken|fish|meat|beef|pork|egg|milk|cheese|butter', case=False,
                                                 na=False) == False]
    else:
        return df


if __name__ == "__main__":
    file_path = "C:/Users/DELL/Downloads/archive2/recipes.csv"  # Updated path to dataset
    df = load_data(file_path)
    nlp = spacy.load("en_core_web_sm")

    print("Options:")
    print("1. Search for a recipe by ingredients")
    print("2. Get a random recipe")
    print("3. Filter recipes by dietary preference")
    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        user_input = input("Enter main ingredients (comma-separated): ")
        ingredients = extract_ingredients(user_input, nlp)
        print(generate_recipe(ingredients, df))
    elif choice == "2":
        print(get_random_recipe(df))
    elif choice == "3":
        diet_type = input("Enter diet type (Vegetarian/Vegan/Non-Vegetarian): ")
        filtered_df = filter_by_diet(df, diet_type)
        print(get_random_recipe(filtered_df) if not filtered_df.empty else "No recipes available for this diet.")
    else:
        print("Invalid choice. Please try again.")