import os
from typing import List, Tuple
from pydub import AudioSegment
from pydub.silence import split_on_silence
from moviepy.editor import VideoFileClip
import numpy as np
import wave
import struct

class AudioExtractor:
    """Class for extracting and processing audio from video files."""
    
    def __init__(self, min_silence_len=500, silence_thresh=-40, keep_silence=300):
        """
        Initialize the AudioExtractor with silence detection parameters.
        
        Args:
            min_silence_len (int): Minimum length of silence (in ms) to be considered a pause.
            silence_thresh (int): Silence threshold in dBFS.
            keep_silence (int): Amount of silence to keep (in ms) around each segment.
        """
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.keep_silence = keep_silence

    def extract_audio(self, video_path: str, output_path: str) -> str:
        """Extract audio from video file using moviepy."""
        try:
            # Ensure input file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Load video and extract audio
            print(f"Loading video: {video_path}")
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                raise ValueError("No audio track found in video file")
                
            audio = video.audio

            # Write audio to WAV file
            print(f"Extracting audio to: {output_path}")
            audio.write_audiofile(output_path, fps=16000, nbytes=2, codec='pcm_s16le')
            
            # Clean up
            video.close()
            audio.close()

            return output_path

        except Exception as e:
            raise RuntimeError(f"Error extracting audio: {str(e)}")

    def process_video(self, video_path: str, output_dir: str) -> List[Tuple[str, float, float]]:
        """Process video file and return audio segments."""
        # Create audio file path
        audio_path = os.path.join(output_dir, "extracted_audio.wav")
        
        # Extract audio
        print(f"Extracting audio from: {video_path}")
        print(f"Output audio path: {audio_path}")
        self.extract_audio(video_path, audio_path)
        
        # Load audio file
        print("Loading audio file for segmentation...")
        audio = AudioSegment.from_wav(audio_path)
        
        # Split audio on silence
        print("Splitting audio on silence...")
        chunks = split_on_silence(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
            keep_silence=self.keep_silence
        )
        
        # Save chunks and create segment info
        segments = []
        current_pos = 0
        
        for i, chunk in enumerate(chunks):
            # Save chunk to file
            chunk_path = os.path.join(output_dir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            
            # Calculate timing
            duration = len(chunk)
            start_time = current_pos
            end_time = start_time + duration
            current_pos = end_time
            
            # Store segment info
            segments.append((chunk_path, start_time / 1000.0, end_time / 1000.0))
        
        return segments


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and segment audio from a video file")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--output", help="Output directory for extracted audio and segments")
    args = parser.parse_args()
    
    extractor = AudioExtractor()
    segments = extractor.process_video(args.video, args.output)
    
    print(f"Processed {len(segments)} segments")
    for i, (segment_path, start_ms, end_ms) in enumerate(segments[:5]):
        print(f"Segment {i}: {segment_path} ({start_ms}ms - {end_ms}ms)")
    if len(segments) > 5:
        print("...")



