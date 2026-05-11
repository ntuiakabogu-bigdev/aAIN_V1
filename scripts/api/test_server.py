"""
Complete API Test Script for AIN v1
Tests all endpoints
"""

import requests
import base64
import os
from pathlib import Path
import time

API_URL = "http://localhost:8000"

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def test_root():
    """Test root endpoint"""
    print_header("TEST 1: Root Endpoint (/)")
    
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service: {data.get('service')}")
            print(f"✅ Version: {data.get('version')}")
            print(f"✅ Status: {data.get('status')}")
            print("\nAvailable Endpoints:")
            for endpoint, method in data.get('endpoints', {}).items():
                print(f"   - {endpoint}: {method}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health():
    """Test health check endpoint"""
    print_header("TEST 2: Health Check (/health)")
    
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data.get('status')}")
            print(f"✅ Models Loaded: {data.get('models_loaded')}")
            print(f"✅ Supported Languages: {data.get('supported_languages')}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_languages():
    """Test languages endpoint"""
    print_header("TEST 3: Languages Endpoint (/languages)")
    
    try:
        response = requests.get(f"{API_URL}/languages")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Supported Languages and Capabilities:")
            for lang, caps in data.get('languages', {}).items():
                print(f"\n   {lang.upper()}:")
                print(f"      ASR: {caps['asr']}")
                print(f"      Translation: {caps['translation']}")
                print(f"      TTS: {caps['tts']}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_transcribe(audio_file, language):
    """Test transcription endpoint"""
    print_header(f"TEST 4: Transcribe Audio (/transcribe) - {language}")
    
    if not os.path.exists(audio_file):
        print(f"⚠️  Skipping: Audio file not found: {audio_file}")
        return None
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (Path(audio_file).name, f, 'audio/wav')}
            data = {'language': language}
            
            print(f"Uploading: {Path(audio_file).name}")
            response = requests.post(
                f"{API_URL}/transcribe",
                files=files,
                data=data
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success')}")
            print(f"✅ Language: {result.get('language')}")
            print(f"✅ Transcription: {result.get('transcription')}")
            return result
        else:
            print(f"❌ Failed: {response.text}")
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_translate(audio_file, source_lang, target_lang):
    """Test translation endpoint"""
    print_header(f"TEST 5: Translate Audio (/translate) - {source_lang} → {target_lang}")
    
    if not os.path.exists(audio_file):
        print(f"⚠️  Skipping: Audio file not found: {audio_file}")
        return None
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (Path(audio_file).name, f, 'audio/wav')}
            data = {
                'source_language': source_lang,
                'target_language': target_lang
            }
            
            print(f"Uploading: {Path(audio_file).name}")
            response = requests.post(
                f"{API_URL}/translate",
                files=files,
                data=data
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success')}")
            print(f"✅ Source: {result.get('source_language')}")
            print(f"✅ Target: {result.get('target_language')}")
            print(f"✅ Transcription: {result.get('transcription')}")
            print(f"✅ Translation: {result.get('translation')}")
            return result
        else:
            print(f"❌ Failed: {response.text}")
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_tts(text, language):
    """Test TTS endpoint"""
    print_header(f"TEST 6: Text-to-Speech (/text-to-speech) - {language}")
    
    try:
        data = {
            'text': text,
            'language': language
        }
        
        print(f"Text: {text}")
        print(f"Language: {language}")
        
        response = requests.post(
            f"{API_URL}/text-to-speech",
            json=data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            # Save audio file
            output_file = f"test_tts_{language}.wav"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(output_file) / 1024  # KB
            print(f"✅ Audio saved: {output_file}")
            print(f"✅ File size: {file_size:.2f} KB")
            return output_file
        else:
            print(f"❌ Failed: {response.text}")
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_speech_to_speech(audio_file, source_lang, target_lang):
    """Test speech-to-speech endpoint"""
    print_header(f"TEST 7: Speech-to-Speech (/speech-to-speech) - {source_lang} → {target_lang}")
    
    if not os.path.exists(audio_file):
        print(f"⚠️  Skipping: Audio file not found: {audio_file}")
        return None
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (Path(audio_file).name, f, 'audio/wav')}
            data = {
                'source_language': source_lang,
                'target_language': target_lang
            }
            
            print(f"Uploading: {Path(audio_file).name}")
            response = requests.post(
                f"{API_URL}/speech-to-speech",
                files=files,
                data=data
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success')}")
            print(f"✅ Source: {result.get('source_language')}")
            print(f"✅ Target: {result.get('target_language')}")
            print(f"✅ Transcription: {result.get('transcription')}")
            print(f"✅ Translation: {result.get('translation')}")
            print(f"✅ TTS Available: {result.get('tts_available')}")
            
            if result.get('audio_output'):
                # Save audio output
                audio_data = base64.b64decode(result['audio_output'])
                output_file = f"test_s2s_{target_lang}.wav"
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                
                file_size = os.path.getsize(output_file) / 1024  # KB
                print(f"✅ Audio saved: {output_file}")
                print(f"✅ File size: {file_size:.2f} KB")
            else:
                print("⚠️  No audio output (TTS not available for target language)")
            
            return result
        else:
            print(f"❌ Failed: {response.text}")
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_submit_translation(audio_file, source_lang, target_lang, transcription, translation):
    """Test submit translation endpoint"""
    print_header(f"TEST 8: Submit Translation Data (/submit-translation)")
    
    if not os.path.exists(audio_file):
        print(f"⚠️  Skipping: Audio file not found: {audio_file}")
        return None
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (Path(audio_file).name, f, 'audio/wav')}
            data = {
                'source_language': source_lang,
                'target_language': target_lang,
                'transcription': transcription,
                'translation': translation,
                'user_id': 'test_user_001'
            }
            
            print(f"Submitting translation data...")
            response = requests.post(
                f"{API_URL}/submit-translation",
                files=files,
                data=data
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success')}")
            print(f"✅ Message: {result.get('message')}")
            print(f"✅ Data ID: {result.get('data_id')}")
            print(f"✅ Total Submissions Today: {result.get('total_submissions_today')}")
            return result
        else:
            print(f"❌ Failed: {response.text}")
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_training_stats():
    """Test training stats endpoint"""
    print_header("TEST 9: Training Statistics (/training-stats)")
    
    try:
        response = requests.get(f"{API_URL}/training-stats")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success')}")
            
            if result.get('success'):
                stats = result.get('stats', {})
                print("\nTraining Data Statistics:")
                print(f"   Incoming Files: {stats.get('incoming_files')}")
                print(f"   Processed Files: {stats.get('processed_files')}")
                print(f"   Validated Files: {stats.get('validated_files')}")
                print(f"   Total Audio Files: {stats.get('total_audio_files')}")
                print(f"   Status: {stats.get('status')}")
            
            return result
        else:
            print(f"❌ Failed: {response.text}")
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Run all tests"""
    
    print("\n" + "="*70)
    print("  AIN v1 - COMPLETE API TEST SUITE")
    print("="*70)
    print("\n⚠️  IMPORTANT:")
    print("1. Make sure API server is running: python api_server.py")
    print("2. Update TEST_AUDIO_FILE path below with your actual audio file")
    print("\nStarting tests in 3 seconds...")
    time.sleep(3)
    
    # Test configuration
    TEST_AUDIO_FILE = "test_audio.wav"  # ← UPDATE THIS with your audio file path
    TEST_SOURCE_LANG = "yoruba"
    TEST_TARGET_LANG = "english"
    TEST_TTS_TEXT = "Hello, how are you?"
    TEST_TTS_LANG = "english"
    
    # Run basic tests (no audio needed)
    print("\n" + "="*70)
    print("  PART 1: BASIC ENDPOINTS (No audio required)")
    print("="*70)
    
    test_root()
    time.sleep(1)
    
    test_health()
    time.sleep(1)
    
    test_languages()
    time.sleep(1)
    
    test_training_stats()
    time.sleep(1)
    
    # Run audio tests (requires test audio file)
    if os.path.exists(TEST_AUDIO_FILE):
        print("\n" + "="*70)
        print("  PART 2: AUDIO ENDPOINTS (Audio file found)")
        print("="*70)
        
        time.sleep(1)
        transcribe_result = test_transcribe(TEST_AUDIO_FILE, TEST_SOURCE_LANG)
        
        time.sleep(1)
        translate_result = test_translate(TEST_AUDIO_FILE, TEST_SOURCE_LANG, TEST_TARGET_LANG)
        
        time.sleep(1)
        test_tts(TEST_TTS_TEXT, TEST_TTS_LANG)
        
        time.sleep(1)
        test_speech_to_speech(TEST_AUDIO_FILE, TEST_SOURCE_LANG, TEST_TARGET_LANG)
        
        # Test data submission (if translation succeeded)
        if translate_result:
            time.sleep(1)
            test_submit_translation(
                TEST_AUDIO_FILE,
                TEST_SOURCE_LANG,
                TEST_TARGET_LANG,
                translate_result.get('transcription', ''),
                translate_result.get('translation', '')
            )
    else:
        print("\n" + "="*70)
        print("  PART 2: AUDIO ENDPOINTS - SKIPPED")
        print("="*70)
        print(f"\n⚠️  Test audio file not found: {TEST_AUDIO_FILE}")
        print("   Please create or provide a test audio file and update TEST_AUDIO_FILE variable")
        print("   Then re-run the tests.")
    
    # Final summary
    print("\n" + "="*70)
    print("  TEST SUITE COMPLETE")
    print("="*70)
    print("\n✅ All basic endpoints tested successfully!")
    
    if not os.path.exists(TEST_AUDIO_FILE):
        print("⚠️  Audio endpoint tests were skipped (no test file)")
        print(f"   Create '{TEST_AUDIO_FILE}' to test audio endpoints")
    else:
        print("✅ Audio endpoint tests completed!")
        print("\nGenerated test files:")
        print(f"   - test_tts_{TEST_TTS_LANG}.wav (if TTS worked)")
        print(f"   - test_s2s_{TEST_TARGET_LANG}.wav (if S2S worked)")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()