import os
import numpy as np
import torch
import torchaudio
from typing import List, Tuple, Dict, Union, Optional
from enum import Enum
import speech_recognition as sr
from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import wave
from tqdm import tqdm
import librosa


class LanguageCode(Enum):
    """Supported language codes."""
    ENGLISH = "english"
    HINDI = "hindi"
    TAMIL = "tamil"
    JAPANESE = "japanese"


class SpeechRecognizer:
    """Class for performing speech recognition using various models."""
    
    def __init__(self):
        """Initialize the speech recognizer."""
        self.models = {}
        self.models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(self.models_dir, exist_ok=True)
    
    def _load_vosk_model(self, language: LanguageCode) -> Model:
        """Load Vosk model for the specified language."""
        language_dir = os.path.join(self.models_dir, language.value)
        model = Model(language_dir)
        return model
    
    def _get_model(self, language: Union[str, LanguageCode], model_type: str = "vosk"):
        """
        Get the appropriate model for the specified language.
        
        Args:
            language (Union[str, LanguageCode]): Language to get model for.
            model_type (str): Type of model to use ("vosk" or "sr").
            
        Returns:
            Model: Loaded model.
        """
        if isinstance(language, str):
            language = LanguageCode(language.lower())
        
        model_key = f"{language.value}_{model_type}"
        
        if model_key not in self.models:
            if model_type == "vosk":
                self.models[model_key] = self._load_vosk_model(language)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        return self.models[model_key]
    
    def recognize_with_vosk(self, audio_path: str, language: Union[str, LanguageCode]) -> str:
        """
        Recognize speech in audio file using Vosk.
        
        Args:
            audio_path (str): Path to the audio file.
            language (Union[str, LanguageCode]): Language of the audio.
            
        Returns:
            str: Recognized text.
        """
        if isinstance(language, str):
            language = LanguageCode(language.lower())
        
        model = self._get_model(language, "vosk")
        
        wf = wave.open(audio_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise ValueError("Audio file must be WAV format mono PCM.")
        
        rec = KaldiRecognizer(model, wf.getframerate())
        
        result = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part_result = json.loads(rec.Result())
                if "text" in part_result:
                    result += part_result["text"] + " "
        
        final_result = json.loads(rec.FinalResult())
        if "text" in final_result:
            result += final_result["text"]
        
        return result.strip()
    
    def recognize_with_sr(self, audio_path: str, language: Union[str, LanguageCode]) -> str:
        """
        Recognize speech in audio file using SpeechRecognition library.
        This uses Google Speech Recognition API when available, 
        and falls back to Sphinx for offline recognition.
        
        Args:
            audio_path (str): Path to the audio file.
            language (Union[str, LanguageCode]): Language of the audio.
            
        Returns:
            str: Recognized text.
        """
        if isinstance(language, str):
            language = LanguageCode(language.lower())
        
        # Language code mapping for Google Speech Recognition
        language_codes = {
            LanguageCode.ENGLISH: "en-US",
            LanguageCode.HINDI: "hi-IN",
            LanguageCode.TAMIL: "ta-IN",
            LanguageCode.JAPANESE: "ja-JP"
        }
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            
        try:
            return recognizer.recognize_google(audio_data, language=language_codes[language])
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service. Using Sphinx instead.")
            # Fall back to Sphinx for offline recognition (English only)
            if language == LanguageCode.ENGLISH:
                return recognizer.recognize_sphinx(audio_data)
            else:
                raise ValueError(f"Offline recognition not available for {language.value}")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return ""
    
    def recognize(self, audio_path: str, language: Union[str, LanguageCode], 
                  model_type: str = "auto") -> str:
        """
        Recognize speech in audio file using the best available model.
        
        Args:
            audio_path (str): Path to the audio file.
            language (Union[str, LanguageCode]): Language of the audio.
            model_type (str): Type of model to use:
                - "auto": Automatically choose the best model
                - "vosk": Use Vosk model
                - "sr": Use SpeechRecognition library
            
        Returns:
            str: Recognized text.
        """
        if isinstance(language, str):
            language = LanguageCode(language.lower())
        
        if model_type == "auto":
            # Try to use the best model for the language
            try:
                return self.recognize_with_vosk(audio_path, language)
            except (FileNotFoundError, ImportError, ValueError) as e:
                print(f"Vosk error: {e}")
                pass
            
            try:
                return self.recognize_with_sr(audio_path, language)
            except (ImportError, ValueError) as e:
                print(f"SpeechRecognition error: {e}")
                raise ValueError(f"No working speech recognition model available for {language.value}")
        
        elif model_type == "vosk":
            return self.recognize_with_vosk(audio_path, language)
        
        elif model_type == "sr":
            return self.recognize_with_sr(audio_path, language)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def batch_recognize(self, audio_segments: List[Tuple[str, int, int]], 
                        language: Union[str, LanguageCode], 
                        model_type: str = "auto") -> List[Tuple[str, int, int]]:
        """
        Recognize speech in multiple audio segments.
        
        Args:
            audio_segments (List[Tuple[str, int, int]]): List of audio segments with paths and timestamps.
            language (Union[str, LanguageCode]): Language of the audio.
            model_type (str): Type of model to use.
            
        Returns:
            List[Tuple[str, int, int]]: List of tuples containing:
                - Recognized text
                - Start time in milliseconds
                - End time in milliseconds
        """
        results = []
        
        for audio_path, start_time, end_time in tqdm(audio_segments, desc=f"Recognizing {language} speech"):
            text = self.recognize(audio_path, language, model_type)
            results.append((text, start_time, end_time))
        
        return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Recognize speech in audio file")
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument("--language", default="english", 
                        choices=["english", "hindi", "tamil", "japanese"],
                        help="Language of the audio")
    parser.add_argument("--model", default="auto", 
                        choices=["auto", "vosk", "sr"],
                        help="Model type to use")
    args = parser.parse_args()
    
    recognizer = SpeechRecognizer()
    text = recognizer.recognize(args.audio, args.language, args.model)
    
    print(f"Recognized text: {text}")

