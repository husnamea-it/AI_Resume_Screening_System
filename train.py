
import pandas as pd

# Read dataset
df = pd.read_csv("dataset/Resume/Resume.csv")

print("First 5 rows:")
print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())