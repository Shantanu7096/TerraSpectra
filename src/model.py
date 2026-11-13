import torch
import torch.nn as nn

class SpatialSpectralHybridViT(nn.Module):
    def __init__(self, in_channels: int = 1, reduced_bands: int = 30, patch_size: int = 9, 
                embed_dim: int = 64, num_heads: int = 4, num_transformer_layers: int = 2, 
                num_classes: int = 2):
        super().__init__()
        
        # 3D-CNN Block for capturing fine-grained spatial-spectral gradient shifts
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )

        # Calculate flattened dimension after 3D convolutions
        spatial_dim = patch_size // 2
        bands_dim = reduced_bands // 2
        self.flattened_dim = 16 * spatial_dim * spatial_dim * bands_dim

        # ViT Linear Projection & Transformer Encoder
        self.projection = nn.Linear(self.flattened_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=128, 
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Classification Head (Healthy vs Chemical Stressed/Pre-symptomatic)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (Batch, 1, Bands, Patch_H, Patch_W)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        
        # Flatten to token representation
        x = x.view(x.size(0), -1)
        token = self.projection(x).unsqueeze(1) # (Batch, 1, Embed_Dim)
        
        # ViT Self-Attention
        attn_out = self.transformer(token)
        logits = self.classifier(attn_out.squeeze(1))
        return logits