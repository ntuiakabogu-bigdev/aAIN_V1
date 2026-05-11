"""Check downloaded ASR DATA"""

import os
import json
from datasets import load_from_disk

DATA_DIR = "../../data/raw/asr"
LANGUAGES = ["  english", "yoruba", "hausa", "igbo"]

def check_dataset(dataset_name, language):
    """Check_dataset for a specific language. """

    path= os.system.join(DATA_DIR, f"{dataset_name}_{language}")

    if not os.path.exists(path):
        print(f"  ✗ Dataset directory {path} does not exist.")
        return False
    try:
        #loading dataset from disk
        dataset = load_from_disk(path)

        #loading metadata
        mtadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
           with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        else:
            metadata = {}
            print(f"{language.capitize()} ")
            print(f" Samples: {len(dataset)}")
            print(f" Location: {path}")

            #Show sample
            if len(dataset) >o:
               sample = dataset[0]
               print(f" Sample keys: {list(sample.keys())[:5]}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to load dataset from {path} with error: {e}")
        return False
    
def main():
   print("\n" + "=" * 60)
   print("Checking Downloaded ASR Datasets")
   print("=" * 60)
    
datasets=all_passed = ["common_voice", "fluers"]

for dataset_name in datasets:
   print("f\n{dataset_name.upper()}:")
   for language in LANGUAGES:
       check_dataset(dataset_name, language)

       print("\n'" + "=" * 60)
       for language in LANGUAGES:
              check_dataset(dataset_name, language)

              print("\n" + "=" * 60)
              print("If all datasets show , the downloads were successful.")
              print("=" * 60)

if __name__ == "__main__":
    main()