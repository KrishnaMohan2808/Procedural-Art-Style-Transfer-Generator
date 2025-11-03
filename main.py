import io
import torch
import re
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import warnings
import os

# Suppress annoying PyTorch UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import local files
from transformer_net import TransformerNet
from procedural_art import generate_procedural_image

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STYLES_DIR = "styles"
AVAILABLE_STYLES = {
    "mosaic": "mosaic.pth",
    "udnie": "udnie.pth",
    "rain-princess": "rain-princess.pth",
    "candy": "candy.pth"
}

# --- FASTAPI APP SETUP ---
app = FastAPI(title="Procedural Multi-Style Transfer API")

# --- CORS (Allow frontend to access this API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL MODEL CACHE ---
model_cache = {}


def load_model(style_name: str) -> torch.nn.Module:
    """Load and cache TransformerNet for the requested style."""
    if style_name not in AVAILABLE_STYLES:
        raise HTTPException(status_code=400, detail=f"Style '{style_name}' not available. Choose from: {list(AVAILABLE_STYLES.keys())}")

    model_path = os.path.join(STYLES_DIR, AVAILABLE_STYLES[style_name])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Use cached model if already loaded
    if style_name in model_cache:
        return model_cache[style_name]

    print(f"🎨 Loading style: {style_name} from {model_path} on device {DEVICE}...")
    model = TransformerNet()
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    # Clean up InstanceNorm running stats
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running\_', k):
            del state_dict[k]

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    model_cache[style_name] = model
    print(f"✅ Loaded and cached model: {style_name}")
    return model


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
        "available_styles": list(AVAILABLE_STYLES.keys()),
        "message": "API is running. You can use the /stylize_procedural/ endpoint to apply styles."
    }


# --- STYLIZATION ENDPOINT ---
@app.post("/stylize_procedural/")
async def stylize_procedural(
    content_file: UploadFile = File(..., description="The image to stylize (JPEG/PNG)"),
    style_name: str = Form("mosaic", description="Choose from: mosaic, udnie, rain-princess, candy"),
    noise_octave: int = Form(4, description="Complexity/detail of the procedural style (1-10)"),
    noise_seed: int = Form(1, description="Random seed for the procedural style (1-100)")
):
    try:
        # 1️ Load model dynamically
        model = load_model(style_name)

        # 2️ Read and process content image
        content_bytes = await content_file.read()
        content_image = Image.open(io.BytesIO(content_bytes)).convert("RGB")

        # Resize for consistent model input
        max_size = 512
        if content_image.size != (max_size, max_size):
            content_image = content_image.resize((max_size, max_size))

        # Force integer parameters
        width, height = int(content_image.width), int(content_image.height)
        octave = int(noise_octave)
        seed = int(noise_seed)

        # 3️ Generate procedural art with same dimensions
        noise_image = generate_procedural_image(width, height, octave=octave, seed=seed)

        # 4️ Blend the two images
        noise_array = np.array(noise_image, dtype=np.float32)
        content_array = np.array(content_image, dtype=np.float32)
        blended_array = (0.5 * noise_array + 0.5 * content_array).astype(np.uint8)
        blended_image = Image.fromarray(blended_array)

        # 5️ Stylize
        content_tensor = preprocess_content(blended_image)
        with torch.no_grad():
            output_tensor = model(content_tensor)

        # 6️ Convert back to image
        stylized_image = postprocess_output(output_tensor)
        buf = io.BytesIO()
        stylized_image.save(buf, format="JPEG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={style_name}_stylized.jpg"}
        )

    except Exception as e:
        print(f" Stylization error: {e}")
        raise HTTPException(status_code=500, detail=f"Stylization failed: {e}")
