import os
import datetime
from typing import List, Tuple, Union, TextIO
import pysrt


class SubtitleFormatter:
    """Class for formatting recognized text into subtitle formats."""
    
    @staticmethod
    def ms_to_timestamp(time_in_seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
        
        Args:
            time_in_seconds (float): Time in seconds.
            
        Returns:
            str: Formatted timestamp.
        """
        # Convert seconds to milliseconds and round to nearest integer
        ms_total = int(round(time_in_seconds * 1000))
        
        # Calculate hours, minutes, seconds, and milliseconds
        hours = ms_total // 3600000
        ms_total %= 3600000
        minutes = ms_total // 60000
        ms_total %= 60000
        seconds = ms_total // 1000
        ms = ms_total % 1000
        
        # Format as HH:MM:SS,mmm
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"
    
    @staticmethod
    def create_subtitle_line(index: int, start_time: float, end_time: float, text: str) -> str:
        """
        Create a subtitle line in SRT format.
        
        Args:
            index (int): Subtitle index number.
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
            text (str): Subtitle text.
            
        Returns:
            str: Formatted subtitle line.
        """
        start_timestamp = SubtitleFormatter.ms_to_timestamp(start_time)
        end_timestamp = SubtitleFormatter.ms_to_timestamp(end_time)
        
        return f"{index}\n{start_timestamp} --> {end_timestamp}\n{text}\n"
    
    @staticmethod
    def format_as_srt(recognized_segments: List[Tuple[str, int, int]], output_file: str) -> None:
        """
        Format recognized text segments as SRT file.
        
        Args:
            recognized_segments (List[Tuple[str, int, int]]): List of tuples containing:
                - Recognized text
                - Start time in milliseconds
                - End time in milliseconds
            output_file (str): Path to output SRT file.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (text, start_ms, end_ms) in enumerate(recognized_segments, 1):
                if not text.strip():
                    continue  # Skip empty segments
                
                subtitle_line = SubtitleFormatter.create_subtitle_line(i, start_ms, end_ms, text)
                f.write(subtitle_line + "\n")
        
        print(f"SRT file created: {output_file}")
    
    @staticmethod
    def clean_and_format_srt(recognized_segments: List[Tuple[str, float, float]], 
                             output_file: str,
                             min_duration_ms: int = 500,
                             max_duration_ms: int = 7000,
                             merge_threshold_ms: int = 200) -> None:
        """
        Clean and format recognized text segments as SRT file with improved readability.
        
        Args:
            recognized_segments (List[Tuple[str, float, float]]): List of tuples containing:
                - Recognized text
                - Start time in seconds
                - End time in seconds
            output_file (str): Path to output SRT file.
            min_duration_ms (int): Minimum duration for a subtitle in milliseconds.
            max_duration_ms (int): Maximum duration for a subtitle in milliseconds.
            merge_threshold_ms (int): Threshold for merging nearby segments in milliseconds.
        """
        # Convert all times to milliseconds for processing
        segments = [(text, int(start * 1000), int(end * 1000)) 
                   for text, start, end in recognized_segments if text.strip()]
        
        if not segments:
            print("No non-empty recognized segments found.")
            return
        
        # Process segments
        merged_segments = []
        current_text = segments[0][0]
        current_start = segments[0][1]
        current_end = segments[0][2]
        
        for text, start, end in segments[1:]:
            # If this segment starts soon after the current one ends
            if start - current_end <= merge_threshold_ms:
                # Merge the segments
                current_text += " " + text
                current_end = end
            else:
                # Ensure minimum duration
                if current_end - current_start < min_duration_ms:
                    current_end = current_start + min_duration_ms
                
                # Ensure maximum duration
                if current_end - current_start > max_duration_ms:
                    # Split into multiple segments if too long
                    words = current_text.split()
                    segments_needed = (current_end - current_start) // max_duration_ms + 1
                    words_per_segment = len(words) // segments_needed
                    
                    for i in range(segments_needed):
                        start_idx = i * words_per_segment
                        end_idx = (i + 1) * words_per_segment if i < segments_needed - 1 else len(words)
                        segment_text = " ".join(words[start_idx:end_idx])
                        
                        segment_duration = max_duration_ms
                        if i == segments_needed - 1:  # Last segment
                            segment_duration = (current_end - current_start) % max_duration_ms
                            if segment_duration < min_duration_ms:
                                segment_duration = min_duration_ms
                        
                        segment_start = current_start + i * max_duration_ms
                        segment_end = segment_start + segment_duration
                        
                        # Convert back to seconds for storage
                        merged_segments.append((segment_text, segment_start / 1000, segment_end / 1000))
                else:
                    # Convert back to seconds for storage
                    merged_segments.append((current_text, current_start / 1000, current_end / 1000))
                
                # Start a new segment
                current_text = text
                current_start = start
                current_end = end
        
        # Add the last segment
        if current_end - current_start < min_duration_ms:
            current_end = current_start + min_duration_ms
        
        if current_end - current_start > max_duration_ms:
            # Split the last segment if too long (similar to above)
            words = current_text.split()
            segments_needed = (current_end - current_start) // max_duration_ms + 1
            words_per_segment = max(1, len(words) // segments_needed)
            
            for i in range(segments_needed):
                start_idx = i * words_per_segment
                end_idx = (i + 1) * words_per_segment if i < segments_needed - 1 else len(words)
                segment_text = " ".join(words[start_idx:end_idx])
                
                segment_duration = max_duration_ms
                if i == segments_needed - 1:  # Last segment
                    segment_duration = (current_end - current_start) % max_duration_ms
                    if segment_duration < min_duration_ms:
                        segment_duration = min_duration_ms
                
                segment_start = current_start + i * max_duration_ms
                segment_end = segment_start + segment_duration
                
                # Convert back to seconds for storage
                merged_segments.append((segment_text, segment_start / 1000, segment_end / 1000))
        else:
            # Convert back to seconds for storage
            merged_segments.append((current_text, current_start / 1000, current_end / 1000))
        
        # Create SRT file
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (text, start_time, end_time) in enumerate(merged_segments, 1):
                subtitle_line = SubtitleFormatter.create_subtitle_line(i, start_time, end_time, text)
                f.write(subtitle_line + "\n")
        
        print(f"Enhanced SRT file created: {output_file}")
    
    @staticmethod
    def create_pysrt_subtitle(recognized_segments: List[Tuple[str, int, int]], output_file: str) -> None:
        """
        Create SRT file using pysrt library for better handling of subtitle formats.
        
        Args:
            recognized_segments (List[Tuple[str, int, int]]): List of tuples containing:
                - Recognized text
                - Start time in milliseconds
                - End time in milliseconds
            output_file (str): Path to output SRT file.
        """
        subtitles = pysrt.SubRipFile()
        
        for i, (text, start_ms, end_ms) in enumerate(recognized_segments, 1):
            if not text.strip():
                continue  # Skip empty segments
            
            # Convert milliseconds to timestamp objects
            start_time = datetime.timedelta(milliseconds=start_ms)
            end_time = datetime.timedelta(milliseconds=end_ms)
            
            # Create a subtitle item
            sub = pysrt.SubRipItem(
                index=i,
                start=pysrt.SubRipTime(milliseconds=start_ms),
                end=pysrt.SubRipTime(milliseconds=end_ms),
                text=text
            )
            
            subtitles.append(sub)
        
        # Save to file
        subtitles.save(output_file, encoding='utf-8')
        print(f"SRT file created with pysrt: {output_file}")
    
    @staticmethod
    def adjust_timing(srt_file: str, offset_ms: int) -> None:
        """
        Adjust timing of all subtitles in a SRT file.
        
        Args:
            srt_file (str): Path to SRT file.
            offset_ms (int): Offset in milliseconds (positive or negative).
        """
        subtitles = pysrt.open(srt_file, encoding='utf-8')
        
        for sub in subtitles:
            sub.start.shift(milliseconds=offset_ms)
            sub.end.shift(milliseconds=offset_ms)
        
        subtitles.save(srt_file, encoding='utf-8')
        print(f"Adjusted timing in SRT file with offset of {offset_ms}ms: {srt_file}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Format subtitles from recognized segments")
    parser.add_argument("--input", help="Input file with recognized segments (if available)")
    parser.add_argument("--output", required=True, help="Output SRT file")
    parser.add_argument("--adjust", type=int, help="Adjust timing by milliseconds")
    args = parser.parse_args()
    
    if args.adjust is not None and os.path.exists(args.output):
        # Adjust timing of existing SRT file
        SubtitleFormatter.adjust_timing(args.output, args.adjust)
    elif args.input:
        # Placeholder for loading recognized segments from a file
        print(f"In a real implementation, this would load segments from {args.input}")
        print(f"For demonstration, creating a sample SRT file")
        
        # Sample recognized segments (text, start_ms, end_ms)
        sample_segments = [
            ("Hello, this is a test.", 0, 2000),
            ("We are creating subtitles.", 2200, 4000),
            ("They will be saved in SRT format.", 4200, 6500),
            ("This is compatible with VLC media player.", 6700, 9000)
        ]
        
        SubtitleFormatter.clean_and_format_srt(sample_segments, args.output)
    else:
        print("Please provide either an input file with recognized segments or an existing SRT file to adjust.")


