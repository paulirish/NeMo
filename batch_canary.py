import argparse
import os
import sys
import glob
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files using NVIDIA Canary model.")
    parser.add_argument("input", help="Path to an audio file or a directory containing audio files.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of files to process in parallel (default: 4)")
    parser.add_argument("--extensions", nargs="+", default=[.wav, .mp3, .flac], help="Audio file extensions to look for in directories.")
    
    args = parser.parse_args()
    
    # 1. Gather files
    input_path = args.input
    files_to_process = []
    
    if os.path.isfile(input_path):
        files_to_process.append(input_path)
    elif os.path.isdir(input_path):
        print(f"📂 Scanning directory: {input_path}")
        for ext in args.extensions:
            # Case-insensitive search simulation
            found = glob.glob(os.path.join(input_path, f"*{ext}"))
            found += glob.glob(os.path.join(input_path, f"*{ext.upper()}"))
            files_to_process.extend(found)
        # Remove duplicates
        files_to_process = list(set(files_to_process))
    else:
        print(f"❌ Error: Input {input_path} not found.")
        sys.exit(1)
        
    if not files_to_process:
        print("⚠️ No audio files found to process.")
        sys.exit(0)
        
    print(f"📝 Found {len(files_to_process)} file(s) to transcribe.")

    # 2. Load Model (The expensive part, done ONLY ONCE)
    print("-" * 60)
    print("🏭 Loading Canary-1b-v2 model... (One-time setup)")
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {map_location}")
    
    try:
        model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-v2', map_location=map_location)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
        
    print("-" * 60)
    
    # 3. Run Inference
    print(f"🚀 Starting transcription (Batch size: {args.batch_size})...")
    
    try:
        # NeMo handles batching internally, which is much more efficient on GPUs
        transcripts = model.transcribe(
            audio=files_to_process,
            batch_size=args.batch_size,
            source_lang='en',
            target_lang='en'
        )
        
        print("-" * 60)
        print("RESULTS")
        print("-" * 60)
        
        # Output results
        for file_path, transcript_obj in zip(files_to_process, transcripts):
            filename = os.path.basename(file_path)
            text = transcript_obj.text
            print(f"📄 {filename}:")
            print(f"   \"{text}\"")
            print("-" * 20)
            
    except Exception as e:
        print(f"❌ Error during transcription: {e}")

if __name__ == "__main__":
    main()
