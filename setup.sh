#!/bin/bash
set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "Downloading Montserrat ExtraBold font..."
mkdir -p fonts
if [ ! -f fonts/Montserrat-ExtraBold.ttf ]; then
    curl -L -o /tmp/montserrat.zip "https://fonts.google.com/download?family=Montserrat"
    unzip -o /tmp/montserrat.zip -d /tmp/montserrat
    cp /tmp/montserrat/static/Montserrat-ExtraBold.ttf fonts/
    rm -rf /tmp/montserrat /tmp/montserrat.zip
    echo "Font downloaded successfully."
else
    echo "Font already exists, skipping download."
fi

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "WARNING: FFmpeg is not installed or not on PATH."
    echo "Please install FFmpeg: https://ffmpeg.org/download.html"
    echo "On Windows with winget:  winget install Gyan.FFmpeg"
    echo "On Windows with choco:   choco install ffmpeg"
    echo ""
fi

echo ""
echo "Setup complete! Run:  streamlit run app.py"
