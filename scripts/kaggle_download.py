import kagglehub
import zipfile
import os

TRANSLATION_DIR = "../../models/pretrained/translation/kaggle"

# Download latest version
try:
    print("\n" + "="*60)
    print("DOWNLOADING: Yoruba Speech Dataset from Kaggle")
    print("="*60)
    print("Size: ~1.5 GB")
    print("Languages: Yoruba")

    # Download dataset
    path = kagglehub.dataset_download("nkowaokwu/ibo-dict")

    # If the dataset is a zip file, extract it manually
    if path.endswith(".zip"):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(TRANSLATION_DIR)

    print("Path to dataset files:", path)
    print(f"Saved to: {TRANSLATION_DIR}")

except Exception as e:
    print(f"Error downloading Yoruba Speech Dataset: {e}")
