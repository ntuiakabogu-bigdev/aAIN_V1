#!/usr/bin/env python3
"""
Test NLLB translation
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = "../../models/pretrained/translation"

# Language codes for NLLB
LANG_CODES = {
    "english": "eng_Latn",
    "yoruba": "yor_Latn",
    "hausa": "hau_Latn",
    "igbo": "ibo_Latn"
}

def load_model():
    """Load translation model"""
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    print("Model loaded")
    return model, tokenizer

def translate(text, source_lang, target_lang, model, tokenizer):
    """Translate text"""
    
    src_code = LANG_CODES[source_lang]
    tgt_code = LANG_CODES[target_lang]
    
    # Set source language
    tokenizer.src_lang = src_code
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate translation
    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
        max_length=200
    )
    
    # Decode
    result = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return result

def test_translations():
    """Test all translation pairs"""
    
    print("\n" + "="*60)
    print("TRANSLATION TEST")
    print("="*60)
    
    model, tokenizer = load_model()
    
    # Test sentences
    tests = [
        # English to Yoruba
        {
            "text": "Hello, how are you?",
            "source": "english",
            "target": "yoruba"
        },
        # Yoruba to English
        {
            "text": "Bawo ni o se wa?",
            "source": "yoruba",
            "target": "english"
        },
        # English to Hausa
        {
            "text": "Good morning",
            "source": "english",
            "target": "hausa"
        },
        # Hausa to English
        {
            "text": "Sannu",
            "source": "hausa",
            "target": "english"
        },
        # English to Igbo
        {
            "text": "Thank you",
            "source": "english",
            "target": "igbo"
        },
        # Igbo to English
        {
            "text": "Kedu",
            "source": "igbo",
            "target": "english"
        }
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"\n--- Test {i} ---")
        print(f"Source ({test['source']}): {test['text']}")
        
        try:
            result = translate(
                test['text'],
                test['source'],
                test['target'],
                model,
                tokenizer
            )
            print(f"Target ({test['target']}): {result}")
            print(" Success")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*60)
    print(" Translation test complete!")
    print("="*60)

if __name__ == "__main__":
    test_translations()