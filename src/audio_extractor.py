import os
import ffmpeg
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
from typing import List, Tuple


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
        
    def extract_audio(self, video_path: str, output_path: str = None) -> str:
        """
        Extract audio from a video file.
        
        Args:
            video_path (str): Path to the video file.
            output_path (str, optional): Path to save the extracted audio.
                If None, saves in the same directory as video with .wav extension.
                
        Returns:
            str: Path to the extracted audio file.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if output_path is None:
            base_path = os.path.splitext(video_path)[0]
            output_path = f"{base_path}.wav"
        
        try:
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec='pcm_s16le', ac=1, ar='16k')
                .run(quiet=True, overwrite_output=True)
            )
            print(f"Audio extracted successfully: {output_path}")
            return output_path
        except ffmpeg.Error as e:
            print(f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
            raise
    
    def segment_audio(self, audio_path: str) -> List[Tuple[AudioSegment, int, int]]:
        """
        Segment audio file based on silence detection.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            List[Tuple[AudioSegment, int, int]]: List of tuples containing:
                - AudioSegment object
                - Start time in milliseconds
                - End time in milliseconds
        """
        print(f"Segmenting audio: {audio_path}")
        
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Split on silence
        chunks = split_on_silence(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
            keep_silence=self.keep_silence
        )
        
        # Calculate start and end times for each chunk
        segments = []
        current_position = 0
        
        for chunk in chunks:
            # Get chunk duration
            chunk_duration = len(chunk)
            
            # Calculate start and end times
            start_time = current_position
            end_time = start_time + chunk_duration
            
            # Add segment to list
            segments.append((chunk, start_time, end_time))
            
            # Update position
            current_position = end_time
        
        print(f"Audio segmented into {len(segments)} chunks")
        return segments
    
    def save_segments(self, segments: List[Tuple[AudioSegment, int, int]], 
                     output_dir: str) -> List[Tuple[str, int, int]]:
        """
        Save audio segments to files.
        
        Args:
            segments (List[Tuple[AudioSegment, int, int]]): List of audio segments.
            output_dir (str): Directory to save the segments.
            
        Returns:
            List[Tuple[str, int, int]]: List of tuples containing:
                - Path to saved segment
                - Start time in milliseconds
                - End time in milliseconds
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_segments = []
        for i, (segment, start_time, end_time) in enumerate(segments):
            segment_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
            segment.export(segment_path, format="wav")
            saved_segments.append((segment_path, start_time, end_time))
        
        return saved_segments
    
    def process_video(self, video_path: str, output_dir: str = None) -> List[Tuple[str, int, int]]:
        """
        Process a video file: extract audio and segment it.
        
        Args:
            video_path (str): Path to the video file.
            output_dir (str, optional): Directory to save extracted audio and segments.
                If None, creates a directory based on the video filename.
                
        Returns:
            List[Tuple[str, int, int]]: List of tuples containing:
                - Path to saved segment
                - Start time in milliseconds
                - End time in milliseconds
        """
        # Create output directory if not provided
        if output_dir is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_audio")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract audio
        audio_path = os.path.join(output_dir, "extracted_audio.wav")
        self.extract_audio(video_path, audio_path)
        
        # Segment audio
        segments = self.segment_audio(audio_path)
        
        # Save segments
        segments_dir = os.path.join(output_dir, "segments")
        saved_segments = self.save_segments(segments, segments_dir)
        
        return saved_segments


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
