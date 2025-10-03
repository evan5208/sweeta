import os
import requests
from pathlib import Path
import hashlib

def download_file(url: str, filepath: str, expected_hash: str = None):
    """Download a file from URL with progress tracking."""
    print(f"Downloading {filepath}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end="", flush=True)
    
    print(f"\nDownloaded {filepath}")
    
    # Verify hash if provided
    if expected_hash:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash != expected_hash:
            raise ValueError(f"Hash mismatch! Expected {expected_hash}, got {file_hash}")
        print("Hash verification passed!")

def ensure_model_weights():
    """Ensure STTN model weights are available."""
    weights_path = "weights/sttn.pth"
    
    if os.path.exists(weights_path):
        print(f"Model weights already exist at {weights_path}")
        return weights_path
    
    print("Model weights not found. Please ensure the STTN model weights are available.")
    print("The weights file should be placed at: weights/sttn.pth")
    
    # For Hugging Face Spaces, you might want to use huggingface_hub to download
    # from a model repository instead
    
    return None

if __name__ == "__main__":
    ensure_model_weights()
