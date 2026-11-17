from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import SpatialSpectralHybridViT
from src.pipeline import HyperspectralCubeProcessor

app = FastAPI(title="TerraSpectra Inference API", version="1.0.0")

# Load processor and model
processor = HyperspectralCubeProcessor(n_components=30)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpatialSpectralHybridViT(reduced_bands=30, patch_size=9, num_classes=2).to(device)
model.eval()

class PredictionRequest(BaseModel):
    file_path: str
    stride: int = 8
    patch_size: int = 9

@app.get("/health")
def health_check():
    return {"status": "active", "device": str(device)}

@app.post("/predict_raster")
def predict_raster(request: PredictionRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="Raster file not found")

    # Ingest and apply PCA
    raw_cube = processor.load_cube(request.file_path)
    reduced_cube = processor.fit_transform_pca(raw_cube)

    h, w, _ = reduced_cube.shape
    risk_map = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    # Patch collection and batch inference
    patches = []
    coords = []
    
    for r, c, patch in processor.extract_patches(reduced_cube, patch_size=request.patch_size, stride=request.stride):
        # Shape formatting: (Bands, H, W) -> PyTorch expects (1, Bands, H, W)
        tensor_patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()
        patches.append(tensor_patch)
        coords.append((r, c))

    if patches:
        batch = torch.stack(patches).to(device) # Shape: (N, 1, Bands, H, W)
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

        for (r, c), prob in zip(coords, probs):
            p_size = request.patch_size
            risk_map[r:min(r + p_size, h), c:min(c + p_size, w)] += prob
            counts[r:min(r + p_size, h), c:min(c + p_size, w)] += 1.0

    counts[counts == 0] = 1.0
    normalized_risk = (risk_map / counts).tolist()

    # Calculate summary metrics
    risk_arr = np.array(normalized_risk)
    flagged_pixels = int(np.sum(risk_arr > 0.5))
    total_acres = round((h * w) * 0.05, 2)
    affected_acres = round(flagged_pixels * 0.05, 2)

    return {
        "status": "success",
        "total_acres_monitored": total_acres,
        "predicted_affected_acres": affected_acres,
        "risk_grid": normalized_risk,
        "risk_percentage": round((affected_acres / total_acres) * 100, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)