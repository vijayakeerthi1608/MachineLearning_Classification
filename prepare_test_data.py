import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
from scipy.io import arff
from sklearn.model_selection import train_test_split

DATASET_CANDIDATES = [
    Path('dry_bean.csv'),
    Path('dry_bean_dataset.csv'),
    Path('Dry_Bean_Dataset.csv'),
]
DATASET_URL = 'https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip'


def load_dataset():
    """Load the Dry Bean dataset from local file or download from UCI."""
    # Try to load from local files
    for path in DATASET_CANDIDATES:
        if path.exists():
            print(f"Loading dataset from {path}")
            return pd.read_csv(path)

    # Download from UCI
    print(f"Downloading dataset from {DATASET_URL}")
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()

    # Handle ZIP file with ARFF content
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        arff_name = next(name for name in archive.namelist() if name.endswith('.arff'))
        with archive.open(arff_name) as arff_file_binary:
            arff_content = arff_file_binary.read().decode('utf-8')
            raw_data, _ = arff.loadarff(io.StringIO(arff_content))
        
        df = pd.DataFrame(raw_data)
        # Decode byte strings
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda value: value.decode() if isinstance(value, (bytes, bytearray)) else value
                )
        return df


# Load the dataset
print("Loading Dry Bean dataset...")
df = load_dataset()
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nClass distribution:\n{df['Class'].value_counts()}")

# Split into train (85%) and test (15%)
print("\nSplitting dataset into 85% train and 15% test...")
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['Class'])

print(f"Train set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Save test set to test_data.csv
output_path = Path('test_data.csv')
test_df.to_csv(output_path, index=False)
print(f"\nTest set saved to {output_path}")
print(f"Test set class distribution:\n{test_df['Class'].value_counts()}")
