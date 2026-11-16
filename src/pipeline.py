import numpy as np
import h5py
from sklearn.decomposition import PCA
from typing import Tuple, Generator

class HyperspectralCubeProcessor:
    def __init__(self, n_components: int = 30):
        """
        Processes multi-band hyperspectral cubes (Height, Width, Bands).
        Reduces 200+ bands to dominant spectral components via PCA.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def generate_synthetic_cube(self, height: int = 128, width: int = 128, bands: int = 224, 
                                save_path: str = "mock_hyperspectral.h5") -> str:
        """Generates realistic synthetic hyperspectral reflectance data cubes."""
        np.random.seed(42)
        # Base vegetation reflectance curve across 400nm - 2500nm
        wavelengths = np.linspace(400, 2500, bands)
        base_spectral_curve = np.exp(-((wavelengths - 850) / 250) ** 2) * 0.6 + 0.1

        cube = np.zeros((height, width, bands), dtype=np.float32)
        for b in range(bands):
            noise = np.random.normal(0, 0.02, (height, width))
            cube[:, :, b] = base_spectral_curve[b] + noise

        # Inject a simulated early fungal outbreak zone (subtle chlorophyll breakdown)
        cube[40:70, 40:70, 50:110] *= 0.72  # Pre-symptomatic NIR degradation

        with h5py.File(save_path, "w") as f:
            f.create_dataset("radiance", data=cube)
            f.attrs["bands"] = bands
            f.attrs["height"] = height
            f.attrs["width"] = width

        return save_path

    def load_cube(self, file_path: str) -> np.ndarray:
        with h5py.File(file_path, "r") as f:
            data = np.array(f["radiance"], dtype=np.float32)
        return data

    def fit_transform_pca(self, cube: np.ndarray) -> np.ndarray:
        """Applies PCA over the spectral dimension to reduce memory footprint."""
        h, w, b = cube.shape
        flat = cube.reshape(-1, b)
        reduced = self.pca.fit_transform(flat)
        self.is_fitted = True
        return reduced.reshape(h, w, self.n_components)

    @staticmethod
    