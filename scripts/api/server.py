"""
AIN v1.2 - Complete REST API
Endpoints: Transcription, Translation, TTS, Speech-to-Speech, Data Collection
Updated: Fine-tuned NLLB model integrated, Igbo TTS enabled
"""
import torch
import soundfile as sf
import librosa
import numpy as np
import base64
import os
import tempfile
from pathlib import Path
import json
import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from speech_to_text_translation import SpeechToTextTranslation
from text_to_speech import TextToSpeech


# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="AIN v1.2 - Complete Translation API",
    description="Speech Translation API for African Languages (Fine-tuned Igbo)",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Igbo TTS now enabled in v1.2
SUPPORTED_LANGUAGES = {
    "english": {"asr": True, "translation": True, "tts": True},
    "yoruba":  {"asr": True, "translation": True, "tts": True},
    "hausa":   {"asr": True, "translation": True, "tts": True},
    "igbo":    {"asr": True, "translation": True, "tts": True},
}

translation_pipeline = None
tts_engine = None


def get_translation_pipeline():
    global translation_pipeline
    if translation_pipeline is None:
        print("Loading translation pipeline...")
        translation_pipeline = SpeechToTextTranslation()
    return translation_pipeline


def get_tts_engine():
    global tts_engine
    if tts_engine is None:
        print("Loading TTS engine...")
        tts_engine = TextToSpeech()
    return tts_engine


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    language: str


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def save_audio_temp(audio_data: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio_data)
    tmp.close()
    return tmp.name


def cleanup(filepath: str):
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
    except:
        pass


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "AIN v1.2 - Translation API",
        "version": "1.2.0",
        "status": "running",
        "whats_new": "Fine-tuned Igbo translation (+12.8 BLEU improvement), Igbo TTS enabled",
        "endpoints": {
            "health": "GET /health",
            "languages": "GET /languages",
            "transcribe": "POST /transcribe",
            "translate": "POST /translate",
            "text_to_speech": "POST /text-to-speech",
            "speech_to_speech": "POST /speech-to-speech",
            "submit_translation": "POST /submit-translation",
            "training_stats": "GET /training-stats"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.2.0",
        "models_loaded": {
            "translation_pipeline": translation_pipeline is not None,
            "tts_engine": tts_engine is not None
        },
        "supported_languages": list(SUPPORTED_LANGUAGES.keys())
    }


@app.get("/languages")
async def get_languages():
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(...)
):
    """Transcribe audio to text (ASR only)"""
    temp_path = None
    try:
        if language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

        audio_data = await file.read()
        temp_path = save_audio_temp(audio_data)

        pipeline = get_translation_pipeline()
        transcription = pipeline.transcribe(temp_path, language)

        if transcription is None:
            raise HTTPException(status_code=500, detail="Transcription failed")

        return {
            "success": True,
            "language": language,
            "transcription": transcription,
            "filename": file.filename
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup(temp_path)


@app.post("/translate")
async def translate_audio(
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...)
):
    """ASR + Translation — returns transcription and translation"""
    temp_path = None
    try:
        if source_language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported source language: {source_language}")
        if target_language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_language}")

        audio_data = await file.read()
        temp_path = save_audio_temp(audio_data)

        pipeline = get_translation_pipeline()
        result = pipeline.process(temp_path, source_language, target_language)

        if result is None:
            raise HTTPException(status_code=500, detail="Translation failed")

        return {
            "success": True,
            "source_language": source_language,
            "target_language": target_language,
            "transcription": result["transcription"],
            "translation": result["translation"],
            "filename": file.filename
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup(temp_path)


@app.post("/text-to-speech")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech — returns audio file"""
    output_path = None
    try:
        if request.language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")

        if not SUPPORTED_LANGUAGES[request.language]["tts"]:
            raise HTTPException(status_code=400, detail=f"TTS not available for {request.language}")

        tts = get_tts_engine()
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"temp_tts_{request.language}_{timestamp}.wav"
        tts.synthesize(request.text, request.language, output_path)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="TTS generation failed")

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"tts_{request.language}.wav"
        )

    except HTTPException:
        raise
    except Exception as e:
        cleanup(output_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speech-to-speech")
async def speech_to_speech(
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...)
):
    """Full pipeline: Audio → Transcribe → Translate → Speak"""
    temp_input = None
    temp_output = None
    try:
        if source_language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported source language: {source_language}")
        if target_language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_language}")

        audio_data = await file.read()
        temp_input = save_audio_temp(audio_data)

        pipeline = get_translation_pipeline()
        result = pipeline.process(temp_input, source_language, target_language)

        if result is None:
            raise HTTPException(status_code=500, detail="Pipeline failed")

        audio_output = None
        audio_filename = None

        if SUPPORTED_LANGUAGES[target_language]["tts"]:
            tts = get_tts_engine()
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            temp_output = f"temp_s2s_{target_language}_{timestamp}.wav"
            tts.synthesize(result["translation"], target_language, temp_output)

            if os.path.exists(temp_output):
                with open(temp_output, "rb") as f:
                    audio_output = base64.b64encode(f.read()).decode("utf-8")
                audio_filename = f"s2s_{target_language}.wav"

        return {
            "success": True,
            "source_language": source_language,
            "target_language": target_language,
            "transcription": result["transcription"],
            "translation": result["translation"],
            "audio_output": audio_output,
            "audio_filename": audio_filename,
            "tts_available": SUPPORTED_LANGUAGES[target_language]["tts"],
            "input_filename": file.filename
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup(temp_input)
        cleanup(temp_output)


@app.post("/submit-translation")
async def submit_translation(
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...),
    transcription: str = Form(...),
    translation: str = Form(...),
    user_id: Optional[str] = Form(None)
):
    """Submit user translation data for future training"""
    try:
        incoming_dir = Path("./data/training/incoming")
        audio_dir = incoming_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"{timestamp}_{source_language}_{user_id or 'anon'}.wav"
        audio_path = audio_dir / audio_filename

        audio_data = await file.read()
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        training_entry = {
            source_language: transcription,
            target_language: translation,
            "metadata": {
                "timestamp": timestamp,
                "audio_file": str(audio_path),
                "user_id": user_id,
                "source": "website",
                "original_filename": file.filename
            }
        }

        today = datetime.datetime.now().strftime("%Y%m%d")
        daily_json = incoming_dir / f"web_data_{today}.json"

        if daily_json.exists():
            with open(daily_json, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
        else:
            existing = []

        existing.append(training_entry)

        with open(daily_json, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "message": "Translation data submitted successfully",
            "data_id": timestamp,
            "total_submissions_today": len(existing)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit data: {str(e)}")


@app.get("/training-stats")
async def get_training_stats():
    """Get statistics about collected training data"""
    try:
        incoming_dir = Path("./data/training/incoming")
        processed_dir = Path("./data/training/processed")
        validated_dir = Path("./data/training/validated")

        return {
            "success": True,
            "stats": {
                "incoming_files": len(list(incoming_dir.glob("*.json"))) if incoming_dir.exists() else 0,
                "processed_files": len(list(processed_dir.glob("*.json"))) if processed_dir.exists() else 0,
                "validated_files": len(list(validated_dir.glob("*.json"))) if validated_dir.exists() else 0,
                "total_audio_files": len(list((incoming_dir / "audio").glob("*.wav"))) if (incoming_dir / "audio").exists() else 0,
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 60)
    print("AIN v1.2 - API Server Starting...")
    print("Fine-tuned Igbo model active")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    print("\nAIN v1.2 API")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")