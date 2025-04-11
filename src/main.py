#!/usr/bin/env python3
import os
import argparse
import time
from typing import List, Tuple, Optional, Dict

from audio_extractor import AudioExtractor
from speech_recognizer import SpeechRecognizer, LanguageCode
from subtitle_formatter import SubtitleFormatter


def generate_subtitles(
    video_path: str,
    language: str,
    output_path: Optional[str] = None,
    model_type: str = "auto",
    min_silence_len: int = 500,
    silence_thresh: int = -40,
    keep_silence: int = 300,
    min_duration_ms: int = 500,
    max_duration_ms: int = 7000,
    merge_threshold_ms: int = 200,
    temp_dir: Optional[str] = None
) -> str:
    """
    Generate subtitles for a video file.
    
    Args:
        video_path (str): Path to the video file.
        language (str): Language of the audio.
        output_path (str, optional): Path to save the SRT file.
            If None, saves in the same directory as video with .srt extension.
        model_type (str): Speech recognition model type to use.
        min_silence_len (int): Minimum length of silence (in ms) to be considered a pause.
        silence_thresh (int): Silence threshold in dBFS.
        keep_silence (int): Amount of silence to keep (in ms) around each segment.
        min_duration_ms (int): Minimum duration for a subtitle in milliseconds.
        max_duration_ms (int): Maximum duration for a subtitle in milliseconds.
        merge_threshold_ms (int): Threshold for merging nearby segments in milliseconds.
        temp_dir (str, optional): Directory to save temporary files.
            If None, creates a directory based on the video filename.
    
    Returns:
        str: Path to the generated SRT file.
    """
    # Validate input file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Determine output path if not provided
    if output_path is None:
        base_path = os.path.splitext(video_path)[0]
        output_path = f"{base_path}.srt"
    
    # Determine temporary directory if not provided
    if temp_dir is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_processing")
    
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Processing video: {video_path}")
    print(f"Language: {language}")
    print(f"Output SRT: {output_path}")
    print(f"Temp directory: {temp_dir}")
    
    start_time = time.time()
    
    # Step 1: Extract audio from video and segment it
    print("\n--- Step 1: Extracting and segmenting audio ---")
    extractor = AudioExtractor(
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    audio_segments = extractor.process_video(video_path, temp_dir)
    
    print(f"Audio extracted and segmented into {len(audio_segments)} chunks")
    
    # Step 2: Perform speech recognition on each segment
    print("\n--- Step 2: Performing speech recognition ---")
    recognizer = SpeechRecognizer()
    recognized_segments = recognizer.batch_recognize(audio_segments, language, model_type)
    
    # Step 3: Format recognized text into SRT file
    print("\n--- Step 3: Formatting subtitles ---")
    SubtitleFormatter.clean_and_format_srt(
        recognized_segments,
        output_path,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        merge_threshold_ms=merge_threshold_ms
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\nSubtitles generated successfully in {processing_time:.2f} seconds")
    print(f"Output file: {output_path}")
    
    return output_path


def list_supported_languages() -> None:
    """Print the list of supported languages."""
    print("Supported languages:")
    for language in LanguageCode:
        print(f"  - {language.value}")


def main() -> None:
    """Main function to parse arguments and generate subtitles."""
    parser = argparse.ArgumentParser(
        description="Generate subtitles for video files in multiple languages"
    )
    
    parser.add_argument("--video", "-v", help="Path to the video file")
    parser.add_argument(
        "--language", "-l", default="english",
        choices=[lang.value for lang in LanguageCode],
        help="Language of the audio"
    )
    parser.add_argument(
        "--output", "-o", 
        help="Path to save the SRT file (default: <video_name>.srt)"
    )
    parser.add_argument(
        "--model", "-m", default="auto",
        choices=["auto", "vosk", "deepspeech", "sr"],
        help="Speech recognition model to use"
    )
    parser.add_argument(
        "--min-silence", type=int, default=500,
        help="Minimum length of silence (in ms) to be considered a pause"
    )
    parser.add_argument(
        "--silence-thresh", type=int, default=-40,
        help="Silence threshold in dBFS"
    )
    parser.add_argument(
        "--keep-silence", type=int, default=300,
        help="Amount of silence to keep (in ms) around each segment"
    )
    parser.add_argument(
        "--min-duration", type=int, default=500,
        help="Minimum duration for a subtitle in milliseconds"
    )
    parser.add_argument(
        "--max-duration", type=int, default=7000,
        help="Maximum duration for a subtitle in milliseconds"
    )
    parser.add_argument(
        "--merge-threshold", type=int, default=200,
        help="Threshold for merging nearby segments in milliseconds"
    )
    parser.add_argument(
        "--temp-dir", 
        help="Directory to save temporary files"
    )
    parser.add_argument(
        "--list-languages", action="store_true",
        help="List supported languages and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_languages:
        list_supported_languages()
        return
    
    if not args.video:
        parser.print_help()
        print("\nError: --video argument is required")
        return
    
    try:
        output_path = generate_subtitles(
            video_path=args.video,
            language=args.language,
            output_path=args.output,
            model_type=args.model,
            min_silence_len=args.min_silence,
            silence_thresh=args.silence_thresh,
            keep_silence=args.keep_silence,
            min_duration_ms=args.min_duration,
            max_duration_ms=args.max_duration,
            merge_threshold_ms=args.merge_threshold,
            temp_dir=args.temp_dir
        )
        
        print(f"\nTo use the generated subtitles in VLC:")
        print(f"1. Open your video file in VLC")
        print(f"2. Go to Subtitles > Add Subtitle File...")
        print(f"3. Select the generated SRT file: {output_path}")
        print(f"4. Enjoy your video with automatically generated subtitles!")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
