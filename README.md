# Speech-to-Text Subtitle Generator

A multilingual speech-to-text subtitle generator for local video files that creates SRT subtitles compatible with VLC media player.

## Supported Languages
- English
- Hindi
- Tamil
- Japanese

## Features
- Audio extraction from video files
- Speech recognition using non-transformer ML/DL architectures
- Speech detection and segmentation
- Automatic subtitle timestamping
- SRT file generation
- Support for multiple languages
- User-friendly web interface with Gradio

## Requirements
- Python 3.8+
- FFmpeg installed on your system
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/stt_subtitle_generator.git
cd stt_subtitle_generator
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Install FFmpeg:
   - Windows: Download from https://ffmpeg.org/download.html and add to PATH
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

## Usage

### Command-Line Interface
```
python src/main.py --video your_video_file.mp4 --language english --output subtitles.srt
```

Available language options: english, hindi, tamil, japanese

### Web Interface (Gradio UI)
For a user-friendly web interface, you can use the Gradio UI:

#### Quick Launch
- **Windows**: Double-click on `run_ui.bat`
- **Linux/Mac**: Run `./run_ui.sh` (you may need to make it executable first with `chmod +x run_ui.sh`)

#### Manual Launch
```
python src/gradio_ui.py
```

This will launch a web server with a user interface where you can:
1. Upload your video file
2. Select language and model options
3. Adjust advanced settings if needed
4. Generate and download the SRT subtitle file

The web interface can be accessed in your browser at http://127.0.0.1:7860

## Project Structure
- `src/`: Source code
  - `audio_extractor.py`: Extracts audio from video files
  - `speech_recognizer.py`: Performs speech recognition
  - `subtitle_formatter.py`: Formats recognized text to SRT
  - `main.py`: Main application script
  - `gradio_ui.py`: Web UI using Gradio
- `models/`: Pre-trained models for different languages
- `data/`: Sample data and intermediate files
- `run_ui.bat`: Windows launcher for the Gradio UI
- `run_ui.sh`: Linux/Mac launcher for the Gradio UI

## How it Works
1. Extracts audio from the video file
2. Segments audio into chunks based on silence detection
3. Performs speech recognition on each chunk using language-specific models
4. Formats the recognized text with timestamps into SRT format
5. Saves the SRT file which can be loaded in VLC or other media players

## License
MIT
