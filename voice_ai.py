"""
Voice AI Module for Interview Coach
- Text-to-Speech (TTS): Speaks questions aloud
- Speech-to-Text (STT): Converts voice answers to text
"""

import io
import tempfile
import os
from typing import Optional, Any

from gtts import gTTS  # type: ignore
import speech_recognition as sr  # type: ignore


class VoiceAI:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.temp_dir = tempfile.gettempdir()
    
    # =========================
    # TEXT-TO-SPEECH (TTS)
    # =========================
    def text_to_speech(self, text: str, lang: str = "en") -> Optional[bytes]:
        """
        Convert text to speech audio bytes.
        Returns MP3 audio as bytes for playback.
        """
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.read()
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
    
    def text_to_speech_file(self, text: str, filename: str = "question.mp3") -> Optional[str]:
        """
        Convert text to speech and save as file.
        Returns the file path.
        """
        try:
            filepath = os.path.join(self.temp_dir, filename)
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(filepath)
            return filepath
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
    
    # =========================
    # SPEECH-TO-TEXT (STT)
    # =========================
    def speech_to_text_from_audio(self, audio_bytes: bytes) -> str:
        """
        Convert audio bytes to text using Google Speech Recognition.
        """
        try:
            # Save audio bytes to temp file
            temp_path = os.path.join(self.temp_dir, "temp_audio.wav")
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
            
            # Recognize speech
            with sr.AudioFile(temp_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)  # type: ignore
                return text
        except sr.UnknownValueError:
            return "[Could not understand audio]"
        except sr.RequestError as e:
            return f"[Speech recognition error: {e}]"
        except Exception as e:
            return f"[Error: {e}]"
    
    def speech_to_text_from_file(self, filepath: str) -> str:
        """
        Convert audio file to text.
        """
        try:
            with sr.AudioFile(filepath) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)  # type: ignore
                return text
        except sr.UnknownValueError:
            return "[Could not understand audio]"
        except sr.RequestError as e:
            return f"[Speech recognition error: {e}]"
        except Exception as e:
            return f"[Error: {e}]"


# =========================
# WHISPER-BASED STT (OPTIONAL - MORE ACCURATE)
# =========================
class WhisperVoiceAI(VoiceAI):
    """
    Enhanced Voice AI using OpenAI Whisper for better accuracy.
    Requires: pip install openai-whisper
    """
    
    def __init__(self, model_size: str = "base"):
        super().__init__()
        self.whisper_model: Any = None
        self.model_size = model_size
    
    def _load_whisper(self) -> bool:
        """Load Whisper model lazily."""
        if self.whisper_model is None:
            try:
                import whisper  # type: ignore
                self.whisper_model = whisper.load_model(self.model_size)
            except ImportError:
                print("Whisper not installed. Run: pip install openai-whisper")
                return False
        return True
    
    def speech_to_text_whisper(self, audio_path: str) -> str:
        """
        Convert audio to text using Whisper.
        More accurate than Google Speech Recognition.
        """
        if not self._load_whisper() or self.whisper_model is None:
            return self.speech_to_text_from_file(audio_path)
        
        try:
            result = self.whisper_model.transcribe(audio_path)
            return str(result["text"]).strip()
        except Exception as e:
            print(f"Whisper Error: {e}")
            return self.speech_to_text_from_file(audio_path)


# Singleton instance
voice_ai = VoiceAI()
