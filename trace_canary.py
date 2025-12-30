# To run this script:
# 1. Have 'uv' installed
# 2. Install dependencies manually (I had to, to bypass project-level resolution issues):
#    uv pip install torch viztracer "nemo-toolkit[asr]"
# 4. Run the script without project context:
#    uv run --no-project trace_canary.py

import os
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel
from viztracer import VizTracer

def main():
    ENABLE_TRACER = True

    if ENABLE_TRACER:
        tracer = VizTracer(output_file="canary_trace.json", log_torch=True, min_duration=700)
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
        text = transcript[0].text
        print(f"Transcript: {text}")
        expected_text = "Paracetamol can help reduce fever"
        if expected_text not in text:
            import sys
            print(f"🔴🔴🔴 ERROR: Transcript does not contain expected text: '{expected_text}' 🔴🔴🔴", file=sys.stderr)
            assert expected_text in text
    else:
        print("No transcript generated.")

    if ENABLE_TRACER:
        tracer.stop()
        tracer.save()
        print("Trace saved to canary_trace.json")

        import json
        with open("canary_trace.json", "r") as f:
            data = json.load(f)
        
        # Remove metadata and file_info
        data.pop("viztracer_metadata", None)
        data.pop("file_info", None)

        with open("canary_trace.json", "w") as f:
            json.dump(data, f)
        print("Cleaned canary_trace.json (removed metadata and file_info)")

if __name__ == "__main__":
    main()
