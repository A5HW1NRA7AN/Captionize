#!/usr/bin/env python3
import os
import tempfile
import shutil
import gradio as gr
import time
from pathlib import Path
from typing import Tuple
import subprocess

from audio_extractor import AudioExtractor
from speech_recognizer import SpeechRecognizer, LanguageCode
from subtitle_formatter import SubtitleFormatter
from main import generate_subtitles


# Set up default paths for temp files
TEMP_DIR = os.path.join(tempfile.gettempdir(), "stt_subtitle_generator")
os.makedirs(TEMP_DIR, exist_ok=True)


def process_video(
    video: str,
    language: str,
    model_type: str,
    min_silence_len: int,
    silence_thresh: int,
    keep_silence: int,
    min_duration: int,
    max_duration: int,
    merge_threshold: int,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Process video and generate subtitles."""
    log_messages = []
    def custom_print(message):
        log_messages.append(str(message))
        print(message)

    try:
        # Debug information
        custom_print(f"Input video path: {video}")
        if not os.path.exists(video):
            raise FileNotFoundError(f"Input video file not found: {video}")

        # Create temp directory for this job
        job_id = f"job_{int(time.time())}"
        temp_job_dir = os.path.join(TEMP_DIR, job_id)
        os.makedirs(temp_job_dir, exist_ok=True)
        custom_print(f"Created temp directory: {temp_job_dir}")

        # Get file extension and create video path
        file_ext = os.path.splitext(video)[1]
        if not file_ext:
            file_ext = '.mp4'  # Default extension if none provided
        video_path = os.path.join(temp_job_dir, f"input{file_ext}")
        custom_print(f"Target video path: {video_path}")

        # Copy the uploaded file to temp directory
        custom_print(f"Copying video file...")
        shutil.copy2(video, video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Failed to copy video to: {video_path}")
        custom_print(f"Video file copied successfully")

        # Set up output path
        output_path = os.path.join(temp_job_dir, "output.srt")
        
        try:
            progress(0.1, "Processing video...")
            custom_print(f"Starting processing with settings:")
            custom_print(f"- Video: {video_path}")
            custom_print(f"- Language: {language}")
            custom_print(f"- Model: {model_type}")

            # Step 1: Extract audio from video and segment it (20%)
            progress(0.2, "Step 1: Extracting and segmenting audio...")
            custom_print("\n--- Step 1: Extracting and segmenting audio ---")
            extractor = AudioExtractor(
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence
            )
            audio_segments = extractor.process_video(video_path, temp_job_dir)
            custom_print(f"Audio extracted and segmented into {len(audio_segments)} chunks")
            
            # Step 2: Perform speech recognition on each segment (50%)
            progress(0.5, "Step 2: Performing speech recognition...")
            custom_print("\n--- Step 2: Performing speech recognition ---")
            recognizer = SpeechRecognizer()
            recognized_segments = recognizer.batch_recognize(audio_segments, language, model_type)
            
            # Step 3: Format recognized text into SRT file (80%)
            progress(0.8, "Step 3: Formatting subtitles...")
            custom_print("\n--- Step 3: Formatting subtitles ---")
            SubtitleFormatter.clean_and_format_srt(
                recognized_segments,
                output_path,
                min_duration_ms=min_duration,
                max_duration_ms=max_duration,
                merge_threshold_ms=merge_threshold
            )
            
            progress(1.0, "Subtitle generation complete!")
            custom_print("\nSubtitles generated successfully")
            custom_print(f"Output file: {output_path}")
            
            return output_path, "\n".join(log_messages)
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}\n"
            error_msg += f"Type: {type(e).__name__}\n"
            import traceback
            error_msg += f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
            custom_print(error_msg)
            return None, "\n".join(log_messages)
            
    except Exception as e:
        error_msg = f"Setup error: {str(e)}\n"
        error_msg += f"Type: {type(e).__name__}\n"
        import traceback
        error_msg += f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
        return None, error_msg


def create_ui():
    """Create the Gradio UI interface."""
    
    with gr.Blocks(title="Speech-to-Text Subtitle Generator") as app:
        gr.Markdown("# Speech-to-Text Subtitle Generator")
        
        gr.Markdown("""
        Generate subtitles for video files in multiple languages. Upload a video file, 
        select options, and get a downloadable SRT subtitle file.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input options
                video_input = gr.Video(label="Upload Video File")
                
                with gr.Group():
                    gr.Markdown("### Language and Model Selection")
                    language = gr.Dropdown(
                        choices=[lang.value for lang in LanguageCode],
                        value="english",
                        label="Language"
                    )
                    model_type = gr.Dropdown(
                        choices=["sr", "vosk", "auto"],  # Reordered to make SR first
                        value="sr",  # Default to SR
                        label="Speech Recognition Model (SR=online Google, Vosk=offline)"
                    )
                    
                    gr.Markdown("""
                    Note: 
                    - "sr" uses Google Speech Recognition (requires internet)
                    - "vosk" requires downloading language models
                    - "auto" tries SR first, then Vosk if available
                    """)
                
                with gr.Accordion("Advanced Settings", open=False):
                    # Audio segmentation settings
                    gr.Markdown("#### Audio Segmentation Settings")
                    min_silence_len = gr.Slider(
                        minimum=100, maximum=2000, value=500, step=50,
                        label="Minimum Silence Length (ms)"
                    )
                    silence_thresh = gr.Slider(
                        minimum=-60, maximum=-10, value=-40, step=1,
                        label="Silence Threshold (dBFS)"
                    )
                    keep_silence = gr.Slider(
                        minimum=100, maximum=1000, value=300, step=50,
                        label="Keep Silence Duration (ms)"
                    )
                    
                    # Subtitle formatting settings
                    gr.Markdown("#### Subtitle Formatting Settings")
                    min_duration = gr.Slider(
                        minimum=200, maximum=2000, value=500, step=50,
                        label="Minimum Subtitle Duration (ms)"
                    )
                    max_duration = gr.Slider(
                        minimum=2000, maximum=10000, value=7000, step=100,
                        label="Maximum Subtitle Duration (ms)"
                    )
                    merge_threshold = gr.Slider(
                        minimum=0, maximum=1000, value=200, step=50,
                        label="Merge Threshold for Nearby Segments (ms)"
                    )
                
                generate_btn = gr.Button("Generate Subtitles", variant="primary")
            
            with gr.Column(scale=1):
                # Output
                srt_output = gr.File(label="Generated SRT File")
                logs_output = gr.Textbox(label="Process Logs", lines=15)
        
        # Set up the function call
        generate_btn.click(
            fn=process_video,
            inputs=[
                video_input, language, model_type, 
                min_silence_len, silence_thresh, keep_silence,
                min_duration, max_duration, merge_threshold
            ],
            outputs=[srt_output, logs_output]
        )
    
    return app


if __name__ == "__main__":
    app = create_ui()
    app.queue()  # Enable queuing
    app.launch(share=True)







