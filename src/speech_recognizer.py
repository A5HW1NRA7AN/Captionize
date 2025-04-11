import os
import numpy as np
import torch
import torchaudio
from typing import List, Tuple, Dict, Union, Optional
from enum import Enum
import speech_recognition as sr
from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import deepspeech
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
    """Class for speech recognition in multiple languages."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the SpeechRecognizer.
        
        Args:
            models_dir (str): Directory containing the speech recognition models.
        """
        self.models_dir = models_dir
        self.models = {}
        SetLogLevel(-1)  # Disable Vosk logging
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
    
    def download_model(self, language: Union[str, LanguageCode]):
        """
        Download required model for the specified language.
        
        Args:
            language (Union[str, LanguageCode]): Language to download model for.
        
        Note: This is a placeholder. In a real implementation, you'd download
        models from official sources or include them with your project.
        """
        if isinstance(language, str):
            language = LanguageCode(language.lower())
        
        language_dir = os.path.join(self.models_dir, language.value)
        os.makedirs(language_dir, exist_ok=True)
        
        print(f"[INFO] Model for {language.value} should be downloaded to {language_dir}")
        print(f"[INFO] In a real implementation, this would download the model automatically.")
        print(f"[INFO] For now, please manually download and place appropriate models in this directory.")
        
        # Model download instructions
        if language == LanguageCode.ENGLISH:
            print("Download English model from: https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
            print("or DeepSpeech model from: https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm")
        elif language == LanguageCode.HINDI:
            print("Download Hindi model from: https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip")
        elif language == LanguageCode.TAMIL:
            print("Download Tamil model from: https://alphacephei.com/vosk/models/vosk-model-small-ta-0.5.zip")
        elif language == LanguageCode.JAPANESE:
            print("Download Japanese model from: https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip")
    
    def _load_vosk_model(self, language: Union[str, LanguageCode]) -> Model:
        """
        Load Vosk model for the specified language.
        
        Args:
            language (Union[str, LanguageCode]): Language to load model for.
            
        Returns:
            Model: Loaded Vosk model.
        """
        if isinstance(language, str):
            language = LanguageCode(language.lower())
        
        language_dir = os.path.join(self.models_dir, language.value)
        
        if not os.path.exists(language_dir):
            self.download_model(language)
            raise FileNotFoundError(f"Model for {language.value} not found. Please download it first.")
        
        return Model(language_dir)
    
    def _load_deepspeech_model(self, language: Union[str, LanguageCode] = LanguageCode.ENGLISH) -> deepspeech.Model:
        """
        Load DeepSpeech model (currently only for English).
        
        Args:
            language (Union[str, LanguageCode]): Language to load model for.
            
        Returns:
            deepspeech.Model: Loaded DeepSpeech model.
        """
        if isinstance(language, str):
            language = LanguageCode(language.lower())
        
        if language != LanguageCode.ENGLISH:
            raise ValueError("DeepSpeech currently only supports English")
        
        language_dir = os.path.join(self.models_dir, language.value)
        model_path = os.path.join(language_dir, "deepspeech-0.9.3-models.pbmm")
        
        if not os.path.exists(model_path):
            self.download_model(language)
            raise FileNotFoundError(f"DeepSpeech model not found at {model_path}. Please download it first.")
        
        model = deepspeech.Model(model_path)
        
        # Load scorer if available (improves accuracy)
        scorer_path = os.path.join(language_dir, "deepspeech-0.9.3-models.scorer")
        if os.path.exists(scorer_path):
            model.enableExternalScorer(scorer_path)
        
        return model
    
    def _get_model(self, language: Union[str, LanguageCode], model_type: str = "vosk"):
        """
        Get the appropriate model for the specified language.
        
        Args:
            language (Union[str, LanguageCode]): Language to get model for.
            model_type (str): Type of model to use ("vosk" or "deepspeech").
            
        Returns:
            Model: Loaded model.
        """
        if isinstance(language, str):
            language = LanguageCode(language.lower())
        
        model_key = f"{language.value}_{model_type}"
        
        if model_key not in self.models:
            if model_type == "vosk":
                self.models[model_key] = self._load_vosk_model(language)
            elif model_type == "deepspeech":
                self.models[model_key] = self._load_deepspeech_model(language)
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
    
    def recognize_with_deepspeech(self, audio_path: str) -> str:
        """
        Recognize speech in audio file using DeepSpeech.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            str: Recognized text.
        """
        model = self._get_model(LanguageCode.ENGLISH, "deepspeech")
        
        # Load audio file
        audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        
        # Ensure we have a 16-bit int array for DeepSpeech
        audio = (audio * 32767).astype(np.int16)
        
        # Recognize speech
        return model.stt(audio)
    
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
                - "deepspeech": Use DeepSpeech model (English only)
                - "sr": Use SpeechRecognition library
            
        Returns:
            str: Recognized text.
        """
        if isinstance(language, str):
            language = LanguageCode(language.lower())
        
        if model_type == "auto":
            # Try to use the best model for the language
            if language == LanguageCode.ENGLISH:
                try:
                    return self.recognize_with_deepspeech(audio_path)
                except (FileNotFoundError, ImportError, ValueError) as e:
                    print(f"DeepSpeech error: {e}")
                    pass
            
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
        
        elif model_type == "deepspeech":
            if language != LanguageCode.ENGLISH:
                raise ValueError("DeepSpeech only supports English")
            return self.recognize_with_deepspeech(audio_path)
        
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
                        choices=["auto", "vosk", "deepspeech", "sr"],
                        help="Model type to use")
    args = parser.parse_args()
    
    recognizer = SpeechRecognizer()
    text = recognizer.recognize(args.audio, args.language, args.model)
    
    print(f"Recognized text: {text}")
