# -------------------------------------------------------
# Grace Arena Ministries - Data Analysis Assignment
# Objective: Load and analyze dataset with pandas & matplotlib
# -------------------------------------------------------

# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print("❌ Error loading dataset:", e)

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check dataset structure
print("\nDataset info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# No missing values in Iris, but let's demonstrate cleaning:
df_clean = df.dropna()

# -------------------------------------------------------
# Task 2: Basic Data Analysis
# -------------------------------------------------------

print("\nStatistical Summary:")
print(df_clean.describe())

# Grouping: mean petal length by species
grouped = df_clean.groupby("target")["petal length (cm)"].mean()
print("\nAverage petal length per species:")
print(grouped)

# -------------------------------------------------------
# Task 3: Data Visualization
# -------------------------------------------------------

# Set style
sns.set(style="whitegrid")

# 1. Line Chart – average sepal length per sample index
plt.figure(figsize=(8,5))
plt.plot(df_clean.index, df_clean["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart: Sepal Length per Sample")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart – average petal length per species
plt.figure(figsize=(8,5))
grouped.plot(kind="bar", color=["skyblue", "lightgreen", "salmon"])
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram – distribution of sepal width
plt.figure(figsize=(8,5))
plt.hist(df_clean["sepal width (cm)"], bins=15, color="purple", alpha=0.7)
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot – sepal length vs. petal length
plt.figure(figsize=(8,5))
plt.scatter(df_clean["sepal length (cm)"], df_clean["petal length (cm)"],
            c=df_clean["target"], cmap="viridis", edgecolor="k", s=50)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.colorbar(label="Species")
plt.show()

# -------------------------------------------------------
# Findings / Observations
# -------------------------------------------------------
"""
1. Different species of Iris have distinct petal lengths – a key feature for classification.
2. Sepal width is normally distributed with some variation across species.
3. Sepal length and petal length are positively correlated (as one increases, so does the other).
4. The dataset is clean (no missing values), making it easy to analyze directly.
"""
