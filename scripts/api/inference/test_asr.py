"""
AIN v1 - Test ASR (Speech-to-Text)
Facebook MMS model
"""

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, AutoProcessor
import os

MODEL_DIR = "facebook/mms-1b-all"

# MMS language codes
LANG_CODES = {
    "english": "eng",
    "yoruba": "yor",
    "hausa": "hau",
    "igbo": "ibo"
}

def load_model(): 
   """Load MMS ASR model and processor"""
   print("Loading MMS model from Hugging Face...")

   try:
       #Load directly from HUGGING FACE(with caching)
       model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/mms-1b-all",
        target_lang = "eng"
       )
       processor = AutoProcessor.from_pretrained(
           "facebook/mms-1b-all",
           target_lang = "eng"
       )

       print("Model loaded sucessfully")
       return model, processor
   except Exception as e:
       print(f"Error loading model: {e}")
       return None, None
   
def transcribe_audio(audio_path, language, model, processor):
    """
    Transcribe audio file to text
    
    Args:
        audio_path: Path to audio file (WAV, 16kHz recommended)
        language: Target language ("yoruba", "hausa", "igbo", "english")
        model: Loaded MMS model
        processor: Loaded processor
    
    Returns:
        Transcribed text
    """
    
    try:
        # STEP 1: Load audio
        print(f"\nLoading audio: {audio_path}")
        import soundfile as sf
        import librosa

        speech_array, sampling_rate = sf.read(audio_path) 
        # STEP 2: Resample if needed (MMS expects 16kHz)
        if sampling_rate != 16000:
            print(f"Resampling from {sampling_rate}Hz to 16000Hz")
            speech_array = librosa.resample(
                speech_array,
                orig_sr= sampling_rate,
                target_sr = 16000
            )
            sampling_rate = 16000
        
        # STEP 3: Convert to mono if stereo
        if len(speech_array.shape) > 1:
            print("Converting to mono")
            speech_array = speech_array.mean(axis = 1)
        
        # STEP 4: Flatten to 1D array
        speech_array = speech_array.squeeze()
        

        target_lang = LANG_CODES[language]
        print(f"language:{language.upper()}")
        processor = AutoProcessor.from_pretrained(
            "facebook/mms-1b-all",
            target_lang = target_lang
        )
        model.load_adapter(target_lang)
        # STEP 6: Process audio
        print("Processing audio...")
        inputs = processor(
            speech_array,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # STEP 7: Run inference
        print("Running ASR inference...")
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # STEP 8: Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        print(f"Transcription: {transcription}")
        return transcription
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def create_test_audio():
    """
    Create a simple test audio file
    (In real use, you'll use actual recorded audio)
    """
    print("\nCreating test audio file...")
    
    # Create a simple sine wave (1 second, 440Hz = A note)
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
    
    # Save as WAV
    test_dir = "../../data/test"
    os.makedirs(test_dir, exist_ok=True)
    test_path = os.path.join(test_dir, "test_audio.wav")
    
    torchaudio.save(test_path, waveform, sample_rate)
    print(f"Test audio created: {test_path}")
    
    return test_path

def test_with_sample_urls():
    """
    Test with sample audio files
    You can download these for testing
    """
    print("\n" + "="*60)
    print("SAMPLE AUDIO FOR TESTING")
    print("="*60)
    print("\nYou can test with your own audio files:")
    print("1. Record yourself speaking in Yoruba/Hausa/Igbo")
    print("2. Save as WAV file (16kHz recommended)")
    print("3. Place in: data/test/")
    print("4. Run this script with your file path")
    print("\nOr download samples from:")
    print("- https://commonvoice.mozilla.org/")
    print("- Record using your phone's voice recorder")
    print("="*60)

def main():
    print("\n" + "="*60)
    print("AIN v1 - ASR TEST")
    print("="*60)
    
    # Load model
    model, processor = load_model()
    
    if model is None or processor is None:
        print("\nfailed to load model. Check Day 1 completed correctly.")
        return
    
    print("\n" + "="*60)
    print("TEST OPTIONS")
    print("="*60)
    print("1. Test with your own audio file")
    print("2. Create dummy audio (for testing setup only)")
    print("3. See sample audio sources")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        audio_path = input("Enter path to audio file: ").strip()
        
        if not os.path.exists(audio_path):
            print(f" File not found: {audio_path}")
            return
        
        # Ask for language
        print("\nAvailable languages:")
        for lang in LANG_CODES.keys():
            print(f"  - {lang}")
        
        language = input("Enter language: ").strip().lower()
        
        if language not in LANG_CODES:
            print(f" Invalid language: {language}")
            return
        
        # Transcribe
        result = transcribe_audio(audio_path, language, model, processor)
        
        if result:
            print("\n" + "="*60)
            print("ASR TEST SUCCESSFUL!")
            print("="*60)
            print(f"Transcription: {result}")
        
    elif choice == "2":
        test_path = create_test_audio()
        
        print("\nNote: This is just a test tone, not real speech")
        print("The transcription won't make sense - this tests the pipeline")
        
        result = transcribe_audio(test_path, "english", model, processor)
        
        if result is not None:
            print("\nPipeline works! Now test with real audio.")
        
    elif choice == "3":
        test_with_sample_urls()
    
    else:
        print("Invalid choice")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Test with real Yoruba/Hausa/Igbo audio")
    print("2. Record yourself speaking")
    print("3. Move to Day 4: Integration")
    print("="*60)

if __name__ == "__main__":
    main()