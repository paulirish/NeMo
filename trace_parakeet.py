# To run this script:
# 1. Ensure you have uv installed.
# 2. Refresh credentials if needed: gcert-local
# 3. Install dependencies manually to bypass project-level resolution issues:
#    uv pip install torch viztracer "nemo-toolkit[asr]"
# 4. Run the script without project context:
#    uv run --no-project trace_parakeet.py

import os
import sys
import torch
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict
from viztracer import VizTracer

def main():
    ENABLE_TRACER = True
    
    if ENABLE_TRACER:
        # Use min_duration to filter out noise, especially from torch internals
        tracer = VizTracer(output_file="parakeet_trace.json", log_torch=True, min_duration=100)
        tracer.start()
    
    # Check if GPU is available
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {map_location}")
    
    # Load model
    try:
        print("Loading NVIDIA Parakeet model (nvidia/parakeet-tdt-0.6b-v3)...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3", map_location=map_location)
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load Parakeet model. Error: {e}")
        if ENABLE_TRACER:
            tracer.stop()
        return

    # Disable CUDA graphs to fix Error 35 on RTX 2000e Ada GPU
    print("Disabling CUDA graphs in TDT decoder...")
    dec_cfg = asr_model.cfg.decoding
    with open_dict(dec_cfg.greedy):
        dec_cfg.greedy['use_cuda_graph_decoder'] = False
    asr_model.change_decoding_strategy(dec_cfg)
    print("✓ CUDA graphs disabled successfully")

    audio_path = "tutorials/tts/audio_samples/phonemes_as_input.wav"
    
    if not os.path.exists(audio_path):
        print(f"Audio file {audio_path} not found.")
        if ENABLE_TRACER:
            tracer.stop()
        return

    # Transcribe
    print("Starting transcription...")
    # transcribe() returns a list of transcripts
    transcripts = asr_model.transcribe(
        audio=[audio_path],
        batch_size=1,
    )
    
    if transcripts:
        # For most ASR models, transcribe returns a list of strings
        # but for some it might be a list of Hypothesis objects.
        # We try to handle both.
        result = transcripts[0]
        if hasattr(result, 'text'):
            print(f"Transcript: {result.text}")
        else:
            print(f"Transcript: {result}")
    else:
        print("No transcript generated.")
    
    if ENABLE_TRACER:
        tracer.stop()
        tracer.save()
        print("Trace saved to parakeet_trace.json")

if __name__ == "__main__":
    main()
