import pandas as pd
import json

# Load dataset
df = pd.read_csv("data/dataset.csv", encoding="latin-1")

# Normalize difficulty
def normalize_diff(x):
    x = str(x).lower()
    if "easy" in x:
        return "easy"
    elif "medium" in x:
        return "intermediate"
    elif "hard" in x:
        return "advanced"
    return "intermediate"

df["difficulty"] = df["Difficulty"].apply(normalize_diff)

# Build clean structure
clean_data = []

for _, row in df.iterrows():
    clean_data.append({
        "question": row["Question"],
        "answer": row["Answer"],
        "category": row["Category"],
        "difficulty": row["difficulty"]
    })

# Save
with open("data/clean_dataset.json", "w") as f:
    json.dump(clean_data, f, indent=2)

print("✅ Clean dataset created!")