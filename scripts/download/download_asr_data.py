
"""
AIN v1 - Download Pre-trained Models
Facebook MMS (ASR) + NLLB (Translation)
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Wav2Vec2ForCTC, AutoProcessor
import os

TRANSLATION_DIR = "../../models/pretrained/translation"
ASR_DIR = "../../models/pretrained/asr"

def download_nllb():
    """Download NLLB-200 translation model"""
    print("\n" + "="*60)
    print("DOWNLOADING: NLLB-200 Translation")
    print("="*60)
    print("Size: ~2.5 GB")
    print("Languages: English ↔ Yoruba, Hausa, Igbo")
    
    os.makedirs(TRANSLATION_DIR, exist_ok=True)
    
    try:
        print("\nDownloading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        )
        model.save_pretrained(TRANSLATION_DIR)
        
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        )
        tokenizer.save_pretrained(TRANSLATION_DIR)
        
        print(f"Saved to: {TRANSLATION_DIR}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def download_mms():
    """Download MMS ASR model"""
    print("\n" + "="*60)
    print("DOWNLOADING: MMS-1B ASR")
    print("="*60)
    print("Size: ~3.5 GB")
    print("Languages: English, Yoruba, Hausa, Igbo")
    
    os.makedirs(ASR_DIR, exist_ok=True)
    
    try:
        print("\nDownloading model...")
        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/mms-1b-all"
        )
        model.save_pretrained(ASR_DIR)
        
        print("Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            "facebook/mms-1b-all"
        )
        processor.save_pretrained(ASR_DIR)
        
        print(f"Saved to: {ASR_DIR}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("AIN v1 - Model Download")
    print("="*60)
    print("\nDownloading:")
    print("1. NLLB-200 (~2.5 GB)")
    print("2. MMS-1B (~3.5 GB)")
    print("\nTotal: ~6 GB")
    print("Time: 30-60 minutes")
    print("="*60)
    
    input("\nPress Enter to start...")
    
    print("\n\n" + "#"*60)
    print("STEP 1/2: TRANSLATION MODEL")
    print("#"*60)
    nllb_ok = download_nllb()
    
    print("\n\n" + "#"*60)
    print("STEP 2/2: ASR MODEL")
    print("#"*60)
    mms_ok = download_mms()
    
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Translation: {'succes' if nllb_ok else 'failed'}")
    print(f"ASR: {'success' if mms_ok else 'failed'}")
   
    if nllb_ok and mms_ok:
        print("\n All models downloaded!")
        print("\nNext: Day 2 - Test the models")
    else:
        print("\n Some downloads failed")

if __name__ == "__main__":
    main()
