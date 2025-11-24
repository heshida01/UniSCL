import pandas as pd

# -----------------------------
# Paths (adapt to your own environment)
# -----------------------------
csv_path = "./feature/test_combined.csv"
importance_path = "./feature/feature_importance_ranking.csv"  # feature importance file
output_path = "./feature/test_top120.csv"                     # output file

# -----------------------------
# Load original data
# -----------------------------
df = pd.read_csv(csv_path)
print("✅ Original data shape:", df.shape)
print("✅ Original columns (first 10):", df.columns[:10].tolist())

# Load feature importance ranking
fi = pd.read_csv(importance_path)
print("✅ Feature importance file loaded:", fi.shape)

# Get top 120 features
top120_features = fi.head(120)["feature"].tolist()
print(f"✅ Number of top features: {len(top120_features)}")

# Check whether all top120 features exist in the original CSV
missing = [f for f in top120_features if f not in df.columns]
if len(missing) > 0:
    print("⚠️ The following features are missing in the original CSV:")
    print(missing)
else:
    print("✅ All top 120 features exist in the original data!")

# Keep only 'task', 'label' and the selected top 120 features
required_columns = ["task", "label"] + top120_features
df_selected = df[required_columns]

print("✅ New data shape:", df_selected.shape)
print("✅ New data columns (first 15):", df_selected.columns[:15].tolist())

# -----------------------------
# Save
# -----------------------------
df_selected.to_csv(output_path, index=False)
print(f"✅ Saved to: {output_path}")
