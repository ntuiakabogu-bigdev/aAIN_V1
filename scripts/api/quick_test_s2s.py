"""
Quick Speech-to-Speech Test Script
Simple script to test speech-to-speech translation
"""

import sys
import os
from pathlib import Path

# Add inference directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'inference'))

from inference.complete_pipeline import CompletePipeline

def main():
    print("\n" + "="*70)
    print("  QUICK SPEECH-TO-SPEECH TEST")
    print("="*70)
    
    # Find test audio files
    inference_dir = Path(__file__).parent / "inference"
    test_files = {
        "english": inference_dir / "test_english.wav",
        "yoruba": inference_dir / "test_yoruba.wav",
        "hausa": inference_dir / "test_hausa.wav"
    }
    
    # Show available test files
    print("\nAvailable test audio files:")
    available_files = {}
    for lang, file_path in test_files.items():
        if file_path.exists():
            print(f"  ✅ {lang}: {file_path.name}")
            available_files[lang] = file_path
        else:
            print(f"  ❌ {lang}: {file_path.name} (not found)")
    
    if not available_files:
        print("\n⚠️  No test audio files found!")
        print(f"   Looking in: {inference_dir}")
        print("\n   Please provide an audio file path:")
        audio_path = input("   Audio file path: ").strip().strip('"').strip("'")
        if not Path(audio_path).exists():
            print(f"   ❌ File not found: {audio_path}")
            return
        source_lang = input("   Source language (english/yoruba/hausa/igbo): ").strip().lower()
    else:
        # Let user choose
        print("\nChoose a test file:")
        for i, (lang, file_path) in enumerate(available_files.items(), 1):
            print(f"  {i}. {lang} ({file_path.name})")
        
        choice = input("\nChoice (1-{}): ".format(len(available_files))).strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_files):
                selected_lang = list(available_files.keys())[idx]
                audio_path = available_files[selected_lang]
                source_lang = selected_lang
            else:
                print("Invalid choice")
                return
        except:
            print("Invalid input")
            return
    
    # Get target language
    print("\nAvailable target languages:")
    print("  1. english")
    print("  2. yoruba")
    print("  3. hausa")
    print("  4. igbo (TTS not available)")
    
    target_choice = input("\nTarget language (1-4 or name): ").strip().lower()
    
    lang_map = {
        "1": "english",
        "2": "yoruba", 
        "3": "hausa",
        "4": "igbo",
        "english": "english",
        "yoruba": "yoruba",
        "hausa": "hausa",
        "igbo": "igbo"
    }
    
    target_lang = lang_map.get(target_choice, target_choice)
    
    if target_lang not in ["english", "yoruba", "hausa", "igbo"]:
        print(f"Invalid target language: {target_lang}")
        return
    
    # Initialize pipeline
    print("\n" + "="*70)
    print("  INITIALIZING PIPELINE")
    print("="*70)
    
    try:
        pipeline = CompletePipeline()
    except Exception as e:
        print(f"\n❌ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run speech-to-speech
    print("\n" + "="*70)
    print(f"  RUNNING SPEECH-TO-SPEECH: {source_lang.upper()} → {target_lang.upper()}")
    print("="*70)
    
    try:
        result = pipeline.process_speech_to_speech(
            str(audio_path),
            source_lang,
            target_lang
        )
        
        if result:
            print("\n" + "="*70)
            print("  ✅ SUCCESS!")
            print("="*70)
            print(f"\n📝 Transcription ({source_lang}):")
            print(f"   {result['transcription']}")
            print(f"\n🌍 Translation ({target_lang}):")
            print(f"   {result['translation']}")
            
            if result.get('speech_output'):
                print(f"\n🎵 Audio output saved:")
                print(f"   {result['speech_output']}")
                print(f"\n   Play this file to hear the translation!")
            else:
                print(f"\n⚠️  No audio output (TTS not available for {target_lang})")
        else:
            print("\n❌ Speech-to-speech failed")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()





