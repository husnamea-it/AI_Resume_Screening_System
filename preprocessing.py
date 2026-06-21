import pandas as pd
import re

# Read dataset
df = pd.read_csv("dataset/Resume/Resume.csv")

# Function to clean resume text
def clean_resume(text):
    text = text.lower()                  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove numbers & symbols
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

# Apply cleaning
df["Clean_Resume"] = df["Resume_str"].apply(clean_resume)

# Show original and cleaned text
print("Original Resume:\n")
print(df["Resume_str"][0])

print("\n" + "="*50 + "\n")

print("Cleaned Resume:\n")
print(df["Clean_Resume"][0])