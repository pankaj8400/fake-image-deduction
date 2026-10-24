"""Dual-Branch Hybrid Deepfake Detection Architecture.

Combines:
1. Spatial Domain: ConvNeXt-Tiny backbone for high-level semantic & structural artifacts.
2. Frequency Domain: SRM (Spatial Rich Model) high-pass filtering + CNN for noise & spectral discrepancies.
3. Attention-Based Cross-Domain Fusion: Dynamic weighting between spatial and spectral cues.
4. Classification Head: 4-class prediction (real, ai_generated, faceswap, photoshop).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Tuple


class SRMConv(nn.Module):
    """Spatial Rich Model (SRM) High-Pass Filtering Layer.
    
    Applies 3 standard steganalysis 5x5 filters (Laplacian, Edge, and Star kernels)
    to reveal high-frequency noise fingerprints, pixel tampering, and diffusion artifacts.
    """
    def __init__(self):
        super().__init__()
        # Buffer for the three 5x5 kernels; loaded from checkpoint
        self.register_buffer("kernels", torch.zeros(3, 1, 5, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W)
        Returns:
            Filtered residual tensor of shape (B, 3, H, W)
        """
        # Convert RGB to Luminance (Grayscale) for SRM analysis: Y = 0.2989*R + 0.5870*G + 0.1140*B
        if x.shape[1] == 3:
            gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        else:
            gray = x[:, 0:1]
        
        return nn.functional.conv2d(gray, self.kernels, padding=2)


class FrequencyBranch(nn.Module):
    """Frequency / Spectral Branch analyzing SRM noise residuals."""
    def __init__(self):
        super().__init__()
        self.srm = SRMConv()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 256)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (embedding_256, srm_residuals)."""
        srm_feat = self.srm(x)
        feat = self.features(srm_feat)
        feat_flat = torch.flatten(feat, 1)
        freq_embed = self.fc(feat_flat)
        return freq_embed, srm_feat


class SpatialBranch(nn.Module):
    """Spatial Branch utilizing ConvNeXt-Tiny features."""
    def __init__(self):
        super().__init__()
        convnext = models.convnext_tiny(weights=None)
        self.backbone = nn.Module()
        self.backbone.features = convnext.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns pooled spatial feature embedding of dimension 768."""
        feat_map = self.backbone.features(x)
        spatial_embed = feat_map.mean([-2, -1])  # Global average pooling
        return spatial_embed


class FusionModule(nn.Module):
    """Cross-Attention Guided Fusion between Spatial and Frequency branches."""
    def __init__(self):
        super().__init__()
        self.spatial_projection = nn.Linear(768, 512)
        self.frequency_projection = nn.Linear(256, 512)
        self.attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        self.output_projection = nn.Linear(1024, 512)

    def forward(self, sp_feat: torch.Tensor, freq_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuses spatial and frequency features with dynamic attention weighting."""
        sp_proj = self.spatial_projection(sp_feat)
        freq_proj = self.frequency_projection(freq_feat)
        
        combined = torch.cat([sp_proj, freq_proj], dim=1)
        attn_gate = self.attention(combined)
        
        weighted_sp = sp_proj * attn_gate
        weighted_freq = freq_proj * (1.0 - attn_gate)
        
        weighted_combined = torch.cat([weighted_sp, weighted_freq], dim=1)
        fused = self.output_projection(weighted_combined)
        return fused, attn_gate


class HybridDeepfakeDetector(nn.Module):
    """Complete End-to-End Hybrid Deepfake Detector."""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.spatial_branch = SpatialBranch()
        self.frequency_branch = FrequencyBranch()
        self.fusion = FusionModule()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning classification logits."""
        sp_feat = self.spatial_branch(x)
        freq_feat, _ = self.frequency_branch(x)
        fused, _ = self.fusion(sp_feat, freq_feat)
        return self.classifier(fused)

    def forward_with_forensics(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Extended forward pass extracting intermediate forensic signals for explainability."""
        sp_feat = self.spatial_branch(x)
        freq_feat, srm_residual = self.frequency_branch(x)
        fused, attn_gate = self.fusion(sp_feat, freq_feat)
        logits = self.classifier(fused)

        # Compute forensic indicators
        with torch.no_grad():
            srm_energy = float(torch.mean(torch.abs(srm_residual)).item())
            laplacian_energy = float(torch.mean(torch.abs(srm_residual[:, 0])).item())
            edge_energy = float(torch.mean(torch.abs(srm_residual[:, 1])).item())
            star_energy = float(torch.mean(torch.abs(srm_residual[:, 2])).item())
            spatial_norm = float(torch.norm(sp_feat, dim=1).mean().item())
            attn_spatial_mean = float(attn_gate.mean().item())

            kernels = {"Laplacian Residual": laplacian_energy, "Edge Residual": edge_energy, "Cross-Star Residual": star_energy}
            dominant_kernel = max(kernels, key=kernels.get)

            # Spatial Noise Inconsistency (Center face zone vs Perimeter background)
            _, _, h, w = srm_residual.shape
            ch_s, ch_e = h // 4, 3 * h // 4
            cw_s, cw_e = w // 4, 3 * w // 4
            center_srm = float(torch.mean(torch.abs(srm_residual[:, :, ch_s:ch_e, cw_s:cw_e])).item())
            
            periphery_mask = torch.ones_like(srm_residual)
            periphery_mask[:, :, ch_s:ch_e, cw_s:cw_e] = 0.0
            periphery_srm = float((torch.sum(torch.abs(srm_residual) * periphery_mask) / (torch.sum(periphery_mask) + 1e-6)).item())
            
            noise_inconsistency = round(abs(center_srm - periphery_srm) / (periphery_srm + 1e-5), 4)

            forensics = {
                "srm_residual_energy": round(srm_energy, 4),
                "dominant_kernel": dominant_kernel,
                "laplacian_residual": round(laplacian_energy, 4),
                "edge_residual": round(edge_energy, 4),
                "star_residual": round(star_energy, 4),
                "spatial_feature_norm": round(spatial_norm, 2),
                "attention_spatial_ratio": round(attn_spatial_mean, 3),
                "attention_frequency_ratio": round(1.0 - attn_spatial_mean, 3),
                "noise_inconsistency": noise_inconsistency,
            }

        return logits, forensics
