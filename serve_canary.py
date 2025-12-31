import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel

# Global model storage
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager that handles startup and shutdown logic.
    This runs exactly once when the server starts.
    """
    print("🏭 Factory opening... Loading Canary-1b-v2 Model...")
    print("   (This one-time setup will take about 30 seconds)")
    
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {map_location}")
    
    try:
        # Load the model into memory
        model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-v2', map_location=map_location)
        ml_models["canary"] = model
        print("✅ Factory open! Model loaded and ready for requests.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise e
        
    yield
    
    # Clean up resources on shutdown
    ml_models.clear()
    print("🏭 Factory closed.")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "ready", "model": "canary-1b-v2"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Endpoint to transcribe an uploaded audio file.
    """
    if "canary" not in ml_models:
        return {"error": "Model not loaded"}

    model = ml_models["canary"]
    
    # NeMo's transcribe method works with file paths, so we save the upload to a temp file
    # Preserve extension as some decoders rely on it
    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".wav" 

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Run inference
        # This part is fast because the model is already in memory
        transcript_obj = model.transcribe(
            audio=[tmp_path], 
            batch_size=1,
            source_lang='en', # Defaulting to English context for now
            target_lang='en'
        )[0]
        
        return {
            "filename": file.filename,
            "text": transcript_obj.text
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    # Run on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
