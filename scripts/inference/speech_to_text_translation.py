"""
AIN v1 - Speech to Text Translation
ASR (Speech Recognition) + Translation Pipeline
v1.2 - Using fine-tuned NLLB model for improved Igbo translation
"""

import torch
from transformers import (
    Wav2Vec2ForCTC,
    AutoProcessor,
    NllbTokenizer,
    AutoModelForSeq2SeqLM
)
import librosa
import numpy as np
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# ── Point to fine-tuned model ──
# Change this path if your AIN_V1 folder is in a different location
TRANSLATION_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "AIN_V1_BACKUP", "ain_v1_2", "nllb_igbo_finetuned", "best_model"
)

# Fallback to HuggingFace if local model not found
if not os.path.exists(TRANSLATION_MODEL_PATH):
    print(f"WARNING: Fine-tuned model not found at {TRANSLATION_MODEL_PATH}")
    print("Falling back to base facebook/nllb-200-distilled-600M")
    TRANSLATION_MODEL_PATH = "facebook/nllb-200-distilled-600M"


class SpeechToTextTranslation:
    """
    Speech Translation Pipeline
    Input: Audio file
    Output: Transcription + Translation
    """
    
    def __init__(self):
        """Initialize models"""
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        print(f"📦 Translation model: {TRANSLATION_MODEL_PATH}")
        
        # Language codes
        self.lang_codes = {
            'yoruba': {'asr': 'yor', 'nllb': 'yor_Latn'},
            'hausa': {'asr': 'hau', 'nllb': 'hau_Latn'},
            'igbo': {'asr': 'ibo', 'nllb': 'ibo_Latn'},
            'english': {'asr': 'eng', 'nllb': 'eng_Latn'}
        }
        
        # Cache for ASR models
        self.asr_models = {}
        self.asr_processors = {}
        
        # Translation model (lazy loading)
        self.translation_tokenizer = None
        self.translation_model = None
        
        print("✅ Pipeline initialized (translation model will load on first use)\n")
    
    def _load_translation_model(self):
        """Load NLLB translation model (lazy loading)"""
        
        if self.translation_model is not None:
            return

        print(f"\n📥 Loading translation model from: {TRANSLATION_MODEL_PATH}")
        
        try:
            print("   Loading NLLB tokenizer...")
            self.translation_tokenizer = NllbTokenizer.from_pretrained(
                TRANSLATION_MODEL_PATH,
                src_lang="eng_Latn"
            )
            
            print("   Loading NLLB model...")
            self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                TRANSLATION_MODEL_PATH
            ).to(self.device)
            
            print("   ✅ Translation model loaded successfully")
            
        except Exception as e:
            print(f"   ❌ Error loading translation model: {e}")
            print("\n   Trying alternative loading method...")
            
            try:
                from transformers import AutoTokenizer
                self.translation_tokenizer = AutoTokenizer.from_pretrained(
                    TRANSLATION_MODEL_PATH,
                    use_fast=False,
                    legacy=False
                )
                self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                    TRANSLATION_MODEL_PATH
                ).to(self.device)
                
                print("   ✅ Translation model loaded (alternative method)")
                
            except Exception as e2:
                print(f"   ❌ Alternative method also failed: {e2}")
                raise Exception("Failed to load translation model with all methods")
    
    def _load_asr_model(self, language):
        """Load ASR model for specific language"""
        
        if language in self.asr_models:
            print(f"   ✅ Using cached ASR model for {language}")
            return self.asr_models[language], self.asr_processors[language]
        
        print(f"   📥 Loading ASR model for {language}...")
        
        asr_code = self.lang_codes[language]['asr']
        
        try:
            processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
            processor.tokenizer.set_target_lang(asr_code)
            
            model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/mms-1b-all",
                target_lang=asr_code,
                ignore_mismatched_sizes=True
            ).to(self.device)
            
            self.asr_models[language] = model
            self.asr_processors[language] = processor
            
            print(f"   ✅ ASR model loaded for {language}")
            return model, processor
            
        except Exception as e:
            print(f"   ❌ Error loading ASR model: {e}")
            raise
    
    def transcribe(self, audio_path, source_language):
        """Step 1: Convert speech to text (ASR)"""
        
        print(f"\n🎤 STEP 1: Transcribing {source_language} audio...")
        
        if not os.path.isfile(audio_path):
            print(f"   ❌ File not found: {audio_path}")
            return None
        
        try:
            print(f"   📂 Loading audio...")
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            
            if len(audio_array) == 0:
                print("   ❌ Audio file is empty")
                return None
            
            print(f"   ✅ Audio loaded: {len(audio_array)} samples")
            
            asr_model, asr_processor = self._load_asr_model(source_language)
            
            print(f"   ⚙️  Processing audio...")
            inputs = asr_processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            
            print(f"   🤖 Running speech recognition...")
            with torch.no_grad():
                outputs = asr_model(**inputs)
                predicted_ids = torch.argmax(outputs.logits, dim=-1)
            
            transcription = asr_processor.batch_decode(predicted_ids)[0].strip()
            
            print(f"   📝 Transcription: {transcription}")
            return transcription
            
        except Exception as e:
            print(f"   ❌ Error in transcription: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def translate(self, text, source_language, target_language):
        """Step 2: Translate text"""
        
        print(f"\n🔄 STEP 2: Translating {source_language} → {target_language}...")
        
        if self.translation_model is None:
            self._load_translation_model()
        
        try:
            src_code = self.lang_codes[source_language]['nllb']
            tgt_code = self.lang_codes[target_language]['nllb']
            
            # Set source language on tokenizer
            self.translation_tokenizer.src_lang = src_code
            
            print(f"   ⚙️  Tokenizing...")
            inputs = self.translation_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            forced_bos_token_id = self.translation_tokenizer.convert_tokens_to_ids(tgt_code)
            
            print(f"   🤖 Running translation...")
            with torch.no_grad():
                generated_tokens = self.translation_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=128,
                    num_beams=5,
                    early_stopping=True
                )
            
            translation = self.translation_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]
            
            print(f"   🌍 Translation: {translation}")
            return translation
            
        except Exception as e:
            print(f"   ❌ Error in translation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process(self, audio_path, source_language, target_language):
        """Complete pipeline: Audio → Transcription → Translation"""
        
        print(f"\n{'='*60}")
        print(f"🎯 PIPELINE: {source_language.upper()} → {target_language.upper()}")
        print(f"{'='*60}")
        
        transcription = self.transcribe(audio_path, source_language)
        
        if transcription is None:
            print("\n❌ Pipeline failed at transcription")
            return None
        
        translation = self.translate(transcription, source_language, target_language)
        
        if translation is None:
            print("\n❌ Pipeline failed at translation")
            return None
        
        return {
            'source_language': source_language,
            'target_language': target_language,
            'transcription': transcription,
            'translation': translation
        }


def main():
    """Test the pipeline"""
    
    print("🚀 AIN v1.2 - Speech to Text Translation Pipeline\n")
    
    try:
        pipeline = SpeechToTextTranslation()
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        return
    
    print("\n" + "="*60)
    audio_path = input("\nAudio file path: ").strip().strip('"').strip("'")
    
    if not os.path.isfile(audio_path):
        print(f"❌ File not found: {audio_path}")
        return
    
    print("\nAvailable languages: english, yoruba, hausa, igbo")
    source_lang = input("Source language (audio): ").strip().lower()
    target_lang = input("Target language (output): ").strip().lower()
    
    try:
        result = pipeline.process(
            audio_path=audio_path,
            source_language=source_lang,
            target_language=target_lang
        )
        
        if result:
            print(f"\n{'='*60}")
            print("✅ RESULT")
            print(f"{'='*60}")
            print(f"Transcription ({source_lang}): {result['transcription']}")
            print(f"Translation ({target_lang}): {result['translation']}")
            print(f"{'='*60}\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")