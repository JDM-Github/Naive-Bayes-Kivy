import os
import pandas as pd

# Define the paths to your dataset folders
base_path = "aclImdb"
pos_path = os.path.join(base_path, "train/pos")
neg_path = os.path.join(base_path, "train/neg")
unsup_path = os.path.join(base_path, "train/unsup")

# Initialize lists to store data
texts = []
sentiments = []

# Function to process files in a folder
def process_folder(folder_path, sentiment_label):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
                texts.append(text)
                sentiments.append(sentiment_label)

# Process positive, negative, and unsupervised reviews
process_folder(pos_path, "positive")
process_folder(neg_path, "negative")
process_folder(unsup_path, "neutral")

# Create a DataFrame
df = pd.DataFrame({
    "text": texts,
    "sentiment": sentiments
})

# Save to CSV
output_path = "aclimdb_sentiment_dataset.csv"
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Dataset created and saved to {output_path}")