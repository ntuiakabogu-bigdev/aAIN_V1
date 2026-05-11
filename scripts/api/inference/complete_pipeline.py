"""
AIN v1.2 - Complete Pipeline
Three modes:
  1. Text → Translate → Speech
  2. Speech-to-Speech (audio file or mic recording)
  3. Real-Time: Speak once → Transcribe → Translate → Speak output → Back to menu
"""
import sys
sys.path.append('.')

from speech_to_text_translation import SpeechToTextTranslation
from text_to_speech import TextToSpeech
import time
import os
import tempfile
from pathlib import Path

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False
    print("sounddevice not installed - microphone input unavailable")
    print("Install with: pip install sounddevice soundfile")


class CompletePipeline:
    """
    AIN v1.2 Complete Pipeline
    Designed for developer integration and automation workflows.
    """

    def __init__(self):
        print("=" * 60)
        print("AIN v1.2 - African Intelligence Network")
        print("=" * 60)
        print("Initializing models...")

        self.translation_pipeline = SpeechToTextTranslation()
        self.tts = TextToSpeech()

        print("Pipeline ready.")
        print("Microphone: " + ("Available" if MIC_AVAILABLE else "Not available (pip install sounddevice)"))
        print("=" * 60)

    # ─────────────────────────────────────────────
    # MODE 1: Text → Translate → Speak
    # ─────────────────────────────────────────────
    def process_text_to_speech(self, text, source_language, target_language, output_path=None):
        """
        Type text in any language → translate → speak in target language.

        Args:
            text: Input text in source_language
            source_language: Language of the input text
            target_language: Language to translate and speak in
            output_path: Optional custom output path for audio file

        Returns:
            dict with input_text, translation, and speech_output path
        """
        print("\n" + "=" * 60)
        print("MODE 1: TEXT → TRANSLATE → SPEECH")
        print("=" * 60)
        start_time = time.time()

        print(f"Input ({source_language}): {text}")

        # Translate if needed
        if source_language == target_language:
            translation = text
            print("(Same language — skipping translation)")
        else:
            print(f"\nTranslating {source_language} → {target_language}...")
            translation = self.translation_pipeline.translate(text, source_language, target_language)
            if not translation:
                print("Translation failed.")
                return None
            print(f"Translation ({target_language}): {translation}")

        # Speak
        if not output_path:
            output_path = f"tts_output_{source_language}_to_{target_language}.wav"

        print(f"\nGenerating speech...")
        speech_output = None
        if self.tts.is_supported(target_language):
            try:
                self.tts.synthesize(translation, target_language, output_path)
                speech_output = output_path
            except Exception as e:
                print(f"TTS failed: {e}")
        else:
            print(f"TTS not available for {target_language} (coming in v1.3)")

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)
        print(f"Input ({source_language}):       {text}")
        print(f"Translation ({target_language}): {translation}")
        print(f"Audio output:               {speech_output or 'N/A'}")
        print(f"Time:                       {round(elapsed, 2)}s")
        print("=" * 60)

        return {
            'mode': 'text-to-speech',
            'source_language': source_language,
            'target_language': target_language,
            'input_text': text,
            'translation': translation,
            'speech_output': speech_output
        }

    # ─────────────────────────────────────────────
    # MODE 2: Speech-to-Speech
    # ─────────────────────────────────────────────
    def process_speech_to_speech(self, audio_path, source_lang, target_lang, cleanup_temp=False, output_path=None):
        """
        Audio file → transcribe → translate → speak output.

        Args:
            audio_path: Path to input audio file
            source_lang: Language spoken in the audio
            target_lang: Language to translate and speak in
            cleanup_temp: Delete input file after processing
            output_path: Optional custom output path for audio

        Returns:
            dict with transcription, translation, and speech_output path
        """
        print("\n" + "=" * 60)
        print("MODE 2: SPEECH-TO-SPEECH")
        print("=" * 60)
        start_time = time.time()

        print(f"Processing: {source_lang} → {target_lang}")

        result = self.translation_pipeline.process(audio_path, source_lang, target_lang)

        if not result:
            print("Pipeline failed at ASR/Translation.")
            if cleanup_temp and os.path.exists(audio_path):
                os.remove(audio_path)
            return None

        result['mode'] = 'speech-to-speech'
        result['speech_output'] = None

        if not output_path:
            output_path = f"s2s_output_{source_lang}_to_{target_lang}.wav"

        print(f"\nGenerating speech output...")
        if self.tts.is_supported(target_lang):
            try:
                self.tts.synthesize(result['translation'], target_lang, output_path)
                result['speech_output'] = output_path
                print("Speech generated successfully.")
            except Exception as e:
                print(f"TTS failed: {e}")
                print("Returning text output only.")
        else:
            print(f"TTS not available for {target_lang} (coming in v1.3)")

        if cleanup_temp and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)
        print(f"Audio input:             {Path(audio_path).name}")
        print(f"Languages:               {source_lang} → {target_lang}")
        print(f"Transcription:           {result['transcription']}")
        print(f"Translation:             {result['translation']}")
        print(f"Audio output:            {result['speech_output'] or 'N/A'}")
        print(f"Time:                    {round(elapsed, 2)}s")
        print("=" * 60)

        return result

    # ─────────────────────────────────────────────
    # MODE 3: Real-Time (one shot)
    # ─────────────────────────────────────────────
    def realtime_translation_once(self, source_lang, target_lang, record_duration=8, sample_rate=16000):
        """
        Record once from mic → transcribe → translate → speak output.
        One complete cycle, then returns.

        Args:
            source_lang: Language you will speak
            target_lang: Language to translate into
            record_duration: How many seconds to record (default 8)
            sample_rate: Audio sample rate (default 16000)

        Returns:
            dict with transcription, translation, speech_output
        """
        if not MIC_AVAILABLE:
            print("Microphone not available. Run: pip install sounddevice soundfile")
            return None

        print("\n" + "=" * 60)
        print("MODE 3: REAL-TIME TRANSLATION")
        print("=" * 60)
        print(f"Languages:      {source_lang} → {target_lang}")
        print(f"Recording:      {record_duration} seconds")
        print("\nPre-loading models...")

        # Pre-load models so there's no delay after recording
        self.translation_pipeline._load_translation_model()
        self.translation_pipeline._load_asr_model(source_lang)

        print(f"\nGet ready — recording starts in 3 seconds...")
        time.sleep(1)
        print("3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(0.5)

        # Record
        print(f"\n🔴 RECORDING ({record_duration}s) — Speak now!\n")
        try:
            audio_data = sd.rec(
                int(record_duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            print("✅ Recording complete.\n")
        except Exception as e:
            print(f"Recording failed: {e}")
            return None

        # Check for silence
        amplitude = np.abs(audio_data).mean()
        if amplitude < 0.005:
            print("No speech detected in recording.")
            return None

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, audio_data, sample_rate)
        tmp.close()

        result = {'transcription': None, 'translation': None, 'speech_output': None}

        try:
            # Step 1: Transcribe
            print("🎤 Transcribing...")
            transcription = self.translation_pipeline.transcribe(tmp.name, source_lang)

            if not transcription or len(transcription.strip()) < 2:
                print("No speech recognized.")
                return None

            result['transcription'] = transcription
            print(f"📝 [{source_lang.upper()}]: {transcription}")

            # Step 2: Translate
            print(f"\n🔄 Translating {source_lang} → {target_lang}...")
            translation = self.translation_pipeline.translate(transcription, source_lang, target_lang)

            if not translation:
                print("Translation failed.")
                return None

            result['translation'] = translation
            print(f"🌍 [{target_lang.upper()}]: {translation}")

            # Step 3: Speak output
            if self.tts.is_supported(target_lang):
                output_path = f"realtime_output_{source_lang}_to_{target_lang}.wav"
                print(f"\n🔊 Generating speech output...")
                try:
                    self.tts.synthesize(translation, target_lang, output_path)
                    result['speech_output'] = output_path
                    print(f"✅ Audio saved: {output_path}")
                except Exception as e:
                    print(f"TTS failed: {e}")
            else:
                print(f"\n⚠️  TTS not available for {target_lang} (coming in v1.3)")

        finally:
            os.unlink(tmp.name)

        print("\n" + "=" * 60)
        print("REAL-TIME TRANSLATION COMPLETE")
        print("=" * 60)
        print(f"Spoke ({source_lang}):        {result['transcription']}")
        print(f"Translation ({target_lang}):  {result['translation']}")
        print(f"Audio output:            {result['speech_output'] or 'N/A'}")
        print("=" * 60)

        return result

    # ─────────────────────────────────────────────
    # Mic recording utility
    # ─────────────────────────────────────────────
    def record_audio_from_mic(self, duration=10, sample_rate=16000):
        if not MIC_AVAILABLE:
            print("Microphone not available.")
            return None
        try:
            print(f"\nRecording for {duration} seconds — speak now!")
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            print("Recording complete.")
            temp_file = "temp_mic_recording.wav"
            sf.write(temp_file, audio_data, sample_rate)
            return temp_file
        except Exception as e:
            print(f"Recording failed: {e}")
            return None


# ─────────────────────────────────────────────
# MAIN - Interactive CLI with menu loop
# ─────────────────────────────────────────────
def show_menu():
    print("\n" + "=" * 60)
    print("AIN v1.2 - SELECT MODE")
    print("=" * 60)
    print("\n1. TEXT → TRANSLATE → SPEECH")
    print("   Type text in any language, translate and speak in target language")
    print("\n2. SPEECH-TO-SPEECH")
    print("   Provide audio file or record from mic, translate and speak output")
    print("\n3. REAL-TIME TRANSLATION")
    print("   Speak into mic once → transcribe → translate → speak output")
    print("\n0. EXIT")
    return input("\nChoose mode (0, 1, 2 or 3): ").strip()


def main():
    print("\n" + "=" * 60)
    print("AIN v1.2 - AFRICAN INTELLIGENCE NETWORK")
    print("=" * 60)

    try:
        pipeline = CompletePipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    valid_langs = ["english", "yoruba", "hausa", "igbo"]

    # ── Main menu loop ──
    while True:
        mode_choice = show_menu()

        if mode_choice == "0":
            print("\nGoodbye.")
            break

        # ── MODE 1 ──
        elif mode_choice == "1":
            print(f"\nAvailable languages: {', '.join(valid_langs)}")
            source_language = input("Source language (language you are typing in): ").strip().lower()
            if source_language not in valid_langs:
                print(f"Invalid language: {source_language}")
                continue

            target_language = input("Target language (language to translate and speak in): ").strip().lower()
            if target_language not in valid_langs:
                print(f"Invalid language: {target_language}")
                continue

            text = input(f"\nEnter text in {source_language}: ").strip()
            if not text:
                print("No text provided.")
                continue

            try:
                pipeline.process_text_to_speech(text, source_language, target_language)
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

        # ── MODE 2 ──
        elif mode_choice == "2":
            print("\n1. Use audio file")
            if MIC_AVAILABLE:
                print("2. Record from microphone")
                input_method = input("\nChoose input method (1 or 2): ").strip()
            else:
                print("2. Record from microphone (unavailable)")
                input_method = "1"

            audio_path = None
            cleanup_temp = False

            if input_method == "1":
                audio_path = input("\nAudio file path: ").strip().strip('"').strip("'")
                if not Path(audio_path).exists():
                    print(f"File not found: {audio_path}")
                    continue
                print(f"File loaded: {Path(audio_path).name}")

            elif input_method == "2" and MIC_AVAILABLE:
                duration_input = input("\nRecording duration in seconds (default 10): ").strip()
                try:
                    duration = int(duration_input) if duration_input else 10
                except:
                    duration = 10
                audio_path = pipeline.record_audio_from_mic(duration=duration)
                if not audio_path:
                    print("Recording failed.")
                    continue
                cleanup_temp = True
            else:
                print("Invalid choice.")
                continue

            print(f"\nAvailable languages: {', '.join(valid_langs)}")
            source_lang = input("Source language (language spoken in audio): ").strip().lower()
            target_lang = input("Target language (language to translate into): ").strip().lower()

            if source_lang not in valid_langs or target_lang not in valid_langs:
                print("Invalid language selection.")
                continue

            try:
                pipeline.process_speech_to_speech(audio_path, source_lang, target_lang, cleanup_temp=cleanup_temp)
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

        # ── MODE 3 ──
        elif mode_choice == "3":
            if not MIC_AVAILABLE:
                print("Microphone not available. Run: pip install sounddevice soundfile")
                continue

            print(f"\nAvailable languages: {', '.join(valid_langs)}")
            source_lang = input("Source language (you will speak in): ").strip().lower()
            target_lang = input("Target language (translation output): ").strip().lower()

            if source_lang not in valid_langs or target_lang not in valid_langs:
                print("Invalid language selection.")
                continue

            duration_input = input("\nHow many seconds to record? (default 8): ").strip()
            try:
                record_duration = int(duration_input) if duration_input else 8
            except:
                record_duration = 8

            try:
                pipeline.realtime_translation_once(source_lang, target_lang, record_duration=record_duration)
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

        else:
            print("Invalid choice. Please enter 0, 1, 2 or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()