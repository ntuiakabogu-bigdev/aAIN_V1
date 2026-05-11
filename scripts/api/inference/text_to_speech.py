"""
AIN v1.2 - Text-to-Speech Module
Using Facebook MMS-TTS
Note: Igbo TTS (mms-tts-ibo) does not exist as a standalone HuggingFace checkpoint.
Igbo TTS will be available in v1.3 via fine-tuning on native speaker audio.
"""

import torch
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile as wavfile
import numpy as np

# Standalone MMS-TTS checkpoints that exist on HuggingFace
TTS_MODELS = {
    "english": "facebook/mms-tts-eng",
    "yoruba":  "facebook/mms-tts-yor",
    "hausa":   "facebook/mms-tts-hau",
}

# Languages with no available TTS model yet
TTS_UNAVAILABLE = {
    "igbo": "Igbo TTS model (mms-tts-ibo) does not exist on HuggingFace. Coming in AIN v1.3."
}


class TextToSpeech:
    """Text-to-Speech using Facebook MMS-TTS"""

    def __init__(self):
        self.models_cache = {}
        print("TTS Module initialized")
        print(f"TTS supported: {list(TTS_MODELS.keys())}")
        print(f"TTS unavailable: {list(TTS_UNAVAILABLE.keys())} (planned for v1.3)")

    def is_supported(self, language: str) -> bool:
        return language in TTS_MODELS

    def load_model(self, language: str):
        """Load TTS model for specific language"""

        if language in TTS_UNAVAILABLE:
            raise ValueError(TTS_UNAVAILABLE[language])

        if language not in TTS_MODELS:
            raise ValueError(f"Language '{language}' not supported for TTS. Supported: {list(TTS_MODELS.keys())}")

        if language in self.models_cache:
            return self.models_cache[language]

        model_name = TTS_MODELS[language]
        print(f"Loading TTS model for {language}: {model_name}")

        try:
            model = VitsModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.models_cache[language] = (model, tokenizer)
            print(f"✅ {language} TTS loaded")
            return model, tokenizer

        except Exception as e:
            print(f"❌ Error loading TTS for {language}: {e}")
            raise

    def synthesize(self, text: str, language: str, output_path: str = None):
        """
        Convert text to speech.

        Args:
            text: Text to speak
            language: Language (english/yoruba/hausa)
            output_path: Path to save .wav file (optional)

        Returns:
            Audio data as numpy array, or None if language unsupported
        """

        if language in TTS_UNAVAILABLE:
            print(f"⚠️  TTS not available for {language}: {TTS_UNAVAILABLE[language]}")
            return None

        model, tokenizer = self.load_model(language)

        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs).waveform

        audio = output.squeeze().cpu().numpy()

        if output_path:
            sample_rate = model.config.sampling_rate
            wavfile.write(output_path, rate=sample_rate, data=audio)
            print(f"Audio saved: {output_path} (sample rate: {sample_rate}Hz)")

        return audio

    def test(self, language: str = "english"):
        """Test TTS with sample text"""

        test_texts = {
            "english": "Hello, how are you?",
            "yoruba":  "Bawo ni o se wa?",
            "hausa":   "Sannu, yaya kake?",
            "igbo":    None  # No TTS available
        }

        if language in TTS_UNAVAILABLE:
            print(f"\n⚠️  Cannot test {language} TTS: {TTS_UNAVAILABLE[language]}")
            return

        text = test_texts.get(language, "Hello")
        print(f"\nTesting TTS for {language}: '{text}'")

        output_path = f"test_tts_{language}.wav"
        result = self.synthesize(text, language, output_path)

        if result is not None:
            print(f"✅ Test complete. Play: {output_path}")
        else:
            print(f"❌ Test failed for {language}")


if __name__ == "__main__":
    tts = TextToSpeech()

    print("\nTTS Test")
    print("=" * 50)
    print("1. English")
    print("2. Yoruba")
    print("3. Hausa")
    print("4. Igbo (unavailable - v1.3)")

    choice = input("\nChoice (1/2/3/4): ").strip()
    lang_map = {"1": "english", "2": "yoruba", "3": "hausa", "4": "igbo"}

    if choice in lang_map:
        tts.test(lang_map[choice])
    else:
        print("Invalid choice")