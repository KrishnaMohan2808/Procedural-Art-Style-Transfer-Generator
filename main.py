import io
import torch
import re
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from starlette.responses import Response 
import warnings
import os

# Suppress annoying PyTorch UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import local files
from transformer_net import TransformerNet 
from procedural_art import generate_procedural_image 

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "styles/mosaic.pth"

# --- FASTAPI APP SETUP ---
app = FastAPI(title="Procedural Style Transfer API")

# --- CORS (Allow frontend to access this API) ---
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODEL ON STARTUP ---
try:
    print(f"Loading model from {MODEL_PATH} on device {DEVICE}...")
    model = TransformerNet()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Skipping load, must download later.")
    
    state_dict = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

    # Clean up InstanceNorm running stats to avoid mismatch
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running\_', k):
            del state_dict[k]
            
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully.")
    
except Exception as e:
    if "skipping load" not in str(e):
        print(f"❌ ERROR: Failed to load model architecture or weights. Full exception: {e}")

# --- IMAGE UTILITIES ---
def preprocess_content(image: Image.Image):
    """Transforms PIL Image into PyTorch tensor for the model."""
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    return content_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

def postprocess_output(output_tensor: torch.Tensor) -> Image.Image:
    """Converts PyTorch output tensor back to PIL Image."""
    img_array = output_tensor[0].clamp(0, 255).cpu().numpy().transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(img_array)

# --- ROOT ENDPOINT (Health check) ---
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "message": "API is running. Model weights may still need to be downloaded."
    }

# --- STYLIZATION ENDPOINT ---
@app.post("/stylize_procedural/")
async def stylize_procedural(
    content_file: UploadFile = File(..., description="The image to stylize (JPEG/PNG)"), 
    noise_octave: int = Form(4, description="Complexity/detail of the procedural style (1-10)"), 
    noise_seed: int = Form(1, description="Random seed for the procedural style (1-100)")
):
    try:
        # Check if model was loaded
        if not hasattr(model, 'load_state_dict'):
            raise RuntimeError("Model weights not loaded. Please complete Step 2.2.")

        # 1️⃣ Read and Process Content Image
        content_bytes = await content_file.read()
        content_image = Image.open(io.BytesIO(content_bytes)).convert("RGB")

        # Resize for consistent model input
        max_size = 512
        if content_image.width != max_size or content_image.height != max_size:
            print(f"ℹ️ Resized content image from {content_image.size} to ({max_size}, {max_size}) for consistency.")
            content_image = content_image.resize((max_size, max_size))

        # ✅ Force parameters to be integers
        width, height = int(content_image.width), int(content_image.height)
        octave = int(noise_octave)
        seed = int(noise_seed)

        # 2️⃣ Generate procedural art with same dimensions
        noise_image = generate_procedural_image(width, height, octave=octave, seed=seed)

        # (Optional) Blend the two for conceptual visualization
        noise_array = np.array(noise_image, dtype=np.float32)
        content_array = np.array(content_image, dtype=np.float32)

        # Ensure same shape
        if noise_array.shape != content_array.shape:
            print(f"⚠️ Resizing noise to match content: {noise_array.shape} -> {content_array.shape}")
            noise_image = noise_image.resize(content_image.size)
            noise_array = np.array(noise_image, dtype=np.float32)

        blended_array = (0.5 * noise_array + 0.5 * content_array).astype(np.uint8)
        blended_image = Image.fromarray(blended_array)

        # 3️⃣ Run stylization
        content_tensor = preprocess_content(blended_image)
        with torch.no_grad():
            output_tensor = model(content_tensor)

        # 4️⃣ Postprocess & Return
        stylized_image = postprocess_output(output_tensor)
        buf = io.BytesIO()
        stylized_image.save(buf, format="JPEG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=stylized_image.jpg"}
        )

    except Exception as e:
        print(f"❌ An error occurred during stylization: {e}")
        raise HTTPException(status_code=500, detail=f"Stylization failed: {e}. Check console for details.")
