import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")

print("ORIGINAL DATASET")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing values safely
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Standardize column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Feature Engineering: Age Group
if "age" in df.columns:
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 30, 50, 100],
        labels=["Child", "Young Adult", "Adult", "Senior"]
    )

# Outlier handling only for selected useful numeric columns
for col in ["age", "fare"]:
    if col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df = df[(df[col] >= lower) & (df[col] <= upper)]

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCLEANED DATASET")
print(df.head())

print("\nFinal Shape:")
print(df.shape)

df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaning Completed Successfully")
print("cleaned_dataset.csv file created.")