# ShortCut

An auto-editor for Reels and Shorts. Point it at a folder of raw clips and it
cuts the silences out, transcribes the audio, burns captions on, drops repeated
phrases where you said the same line twice, and exports. There's a Streamlit UI
and a CLI.

## Prerequisites

- Python 3.10 or newer
- FFmpeg on your PATH. On Windows,
  `winget install Gyan.FFmpeg` or `choco install ffmpeg`
- An Anthropic API key, used to pick out the keywords that get emphasised in the
  captions

## Setup

```bash
git clone <your-fork>
cd ShortCut
```

Put your key in a `.env` in the project root:

```
ANTHROPIC_API_KEY=your-key-here
```

Then either run the setup script, which installs the dependencies, the spaCy
model, and the font:

```bash
bash setup.sh
```

or do it by hand:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Doing it by hand means putting `Montserrat-ExtraBold.ttf` in `fonts/` yourself.
Without it the captions fall back to a system font, and they'll look wrong
rather than break.

## Running it

```bash
streamlit run app.py
```

The UI opens at http://localhost:8501. `cli.py` runs the same pipeline without
the browser.

## The settings that matter

Silence detection is two numbers, and they're the ones worth touching:

| Setting | Default | What it does |
|---|---|---|
| `silence_threshold` | -35 dB | Audio below this counts as silence. Around -28 dB cuts tight, -35 dB sounds natural |
| `min_silence_dur` | 0.4s | A gap has to last this long before it gets cut |
| `silence_padding` | 0.05s | Left on either side of a cut so words don't get clipped |

Cutting at -28 dB gives you the fast-paced edit people expect from Shorts, and
it also clips breaths in a way that sounds airless over a whole minute. -35 dB
leaves the rhythm of how you actually talk.

Presets are saved in `presets.json`, so once you've found the numbers for your
mic and your room you keep them.

Long-form mode skips transcription and caption rendering entirely, which is much
faster when you only want the silences gone.

## GPU

If you have an NVIDIA card with CUDA, transcription runs on the GPU through
faster-whisper and encoding uses NVENC. Detection happens on startup with nothing
to configure, and the header shows you what it found for FFmpeg, CUDA, and
NVENC so you can tell which path you're on before you queue up an hour of
footage.

## Layout

```
app.py            the Streamlit UI
cli.py            the same pipeline from a terminal
core/silence.py   silence detection and cutting
core/captions.py  caption rendering and font handling
core/duplicates.py   removes repeated phrases
utils/
presets.json      your saved settings
fonts/            Montserrat ExtraBold
```

faster-whisper for transcription, spaCy for the language work, Anthropic for
keyword selection, Pillow for the caption rendering.

## Limitations

- English only, because the spaCy model is `en_core_web_sm`.
- Duplicate removal matches phrases, so it catches a line you repeated and not a
  take you'd rather drop for other reasons.
- Captions are burned in. There's no sidecar SRT.
- CPU transcription works and it's slow enough that you'll notice on anything
  longer than a couple of minutes.

## License

MIT. See [LICENSE](LICENSE).

More at [benattanasio.com/lab](https://benattanasio.com/lab).
