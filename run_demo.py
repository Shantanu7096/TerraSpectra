import os
from src.pipeline import HyperspectralCubeProcessor
from src.model import SpatialSpectralHybridViT
import torch

def verify_pipeline():
    print("1. Generating synthetic 224-band hyperspectral cube...")
    proc = HyperspectralCubeProcessor(n_components=30)
    cube_path = proc.generate_synthetic_cube(height=64, width=64, bands=224, save_path="mock_hyperspectral.h5")
    print(f"   Created: {cube_path}")

    print("2. Loading cube and running PCA dimensionality reduction...")
    cube = proc.load_cube(cube_path)
    reduced = proc.fit_transform_pca(cube)
    print(f"   Reduced shape from {cube.shape} -> {reduced.shape}")

    print("3. Validating 3D-CNN + ViT forward pass...")
    model = SpatialSpectralHybridViT(reduced_bands=30, patch_size=9, num_classes=2)
    
    # Extract one sample batch
    for r, c, patch in proc.extract_patches(reduced, patch_size=9, stride=9):
        sample_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            output = model(sample_tensor)
            probs = torch.softmax(output, dim=1)
        print(f"   Forward pass successful! Sample output logits shape: {output.shape}")
        print(f"   Predicted class probabilities: {probs.numpy()}")
        break

    print("\nEnvironment & pipeline ready for Week 1 & Week 2 deliverables!")

if __name__ == "__main__":
    verify_pipeline()  
    
    
    
    
    def verify_pipeline():
    print("1. Generating synthetic 224-band hyperspectral cube...")
    proc = HyperspectralCubeProcessor(n_components=30)
    cube_path = proc.generate_synthetic_cube(height=64, width=64, bands=224, save_path="mock_hyperspectral.h5")
    print(f"   Created: {cube_path}")

    print("2. Loading cube and running PCA dimensionality reduction...")
    cube = proc.load_cube(cube_path)
    reduced = proc.fit_transform_pca(cube)
    print(f"   Reduced shape from {cube.shape} -> {reduced.shape}")

    print("3. Validating 3D-CNN + ViT forward pass...")
    model = SpatialSpectralHybridViT(reduced_bands=30, patch_size=9, num_classes=2)
    
    # Extract one sample batch
    for r, c, patch in proc.extract_patches(reduced, patch_size=9, stride=9):
        sample_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            output = model(sample_tensor)
            probs = torch.softmax(output, dim=1)
        print(f"   Forward pass successful! Sample output logits shape: {output.shape}")
        print(f"   Predicted class probabilities: {probs.numpy()}")
        break

    print("\nEnvironment & pipeline ready for Week 1 & Week 2 deliverables!")

if __name__ == "__main__":
    verify_pipeline()