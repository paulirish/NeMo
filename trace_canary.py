# To run this script:
# 1. Ensure you have uv installed.
# 2. Refresh credentials if needed: gcert-local
# 3. Install dependencies manually to bypass project-level resolution issues:
#    uv pip install torch viztracer "nemo-toolkit[asr]"
# 4. Run the script without project context:
#    uv run --no-project trace_canary.py

import os
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel
from viztracer import VizTracer

def main():
    ENABLE_TRACER = False

    if ENABLE_TRACER:
        tracer = VizTracer(output_file="canary_trace.json")
        tracer.start()

    # Check if GPU is available
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {map_location}")

    # Load model
    # We use canary-1b-v2 as it is the latest and most comprehensive model
    try:
        print("Loading canary-1b-v2 model...")
        canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-v2', map_location=map_location)
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load canary-1b-v2. Error: {e}")
        return

    audio_path = "tutorials/tts/audio_samples/phonemes_as_input.wav"

    if not os.path.exists(audio_path):
        print(f"Audio file {audio_path} not found.")
        return

    # Transcribe
    print("Starting transcription...")
    # The return value of transcribe is a list of objects (Hypothesis or similar) that have a .text attribute
    transcript = canary_model.transcribe(
        audio=[audio_path],
        batch_size=1,
        source_lang='en',
        target_lang='en',
    )

    # Access the text of the first transcript
    if transcript:
        print(f"Transcript: {transcript[0].text}")
    else:
        print("No transcript generated.")

    if ENABLE_TRACER:
        tracer.stop()
        tracer.save()
        print("Trace saved to canary_trace.json")

if __name__ == "__main__":
    main()
