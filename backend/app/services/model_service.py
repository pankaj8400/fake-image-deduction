"""Service managing PyTorch deepfake model lifecycle and inference."""

import io
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from PIL import Image, ImageOps

try:
    from ..config import settings
    from ..models.architecture import HybridDeepfakeDetector
except (ImportError, ValueError):
    try:
        from app.config import settings
        from app.models.architecture import HybridDeepfakeDetector
    except (ImportError, ValueError):
        from config import settings
        from models.architecture import HybridDeepfakeDetector

logger = logging.getLogger(__name__)

# Native aspect ratios standard in modern generative diffusion models (Gemini, Midjourney, DALL-E 3, SDXL)
AI_GENERATIVE_RESOLUTIONS = {
    (1408, 768), (768, 1408),   # Gemini / Imagen 3 native widescreen & portrait
    (1024, 1024),               # DALL-E 3, Midjourney v6, SDXL 1:1
    (1536, 1024), (1024, 1536), # Midjourney v6 3:2, DALL-E 3
    (1792, 1024), (1024, 1792), # DALL-E 3 16:9 & 9:16
    (1344, 768), (768, 1344),   # Midjourney 16:9
    (512, 512), (768, 768)      # Stable Diffusion 1.5 / 2.1
}

# Environmental landscape & scenic categories where a FaceSwap is impossible
LANDSCAPE_SCENE_KEYWORDS = {
    'alp', 'mountain', 'valley', 'cliff', 'lake', 'sea', 'forest', 'desert',
    'volcano', 'promontory', 'geyser', 'sandbar', 'lakeside', 'seashore',
    'ocean', 'landscape', 'plateau', 'wilderness'
}


class ModelService:
    """Production service for loading, preprocessing, and running image inference."""

    _instance: Optional["ModelService"] = None

    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.classes: List[str] = settings.CLASSES
        self.model: Optional[HybridDeepfakeDetector] = None
        self.vit_model: Optional[Any] = None
        self.vit_categories: List[str] = []
        self.vit_transforms: Optional[Any] = None
        self.is_loaded: bool = False
        self.model_path: Optional[Path] = None

        # 1. Standard global view transform (224x224)
        self.transform_std = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 2. Aspect-ratio preserving center crop transform (256 -> 224)
        self.transform_crop = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(settings.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self) -> None:
        """Loads model weights from the configured checkpoint path."""
        resolved_path = settings.get_resolved_model_path()
        self.model_path = resolved_path

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {resolved_path}. "
                f"Please verify hybrid_deepfake_smoke_test.pth is located in the backend directory."
            )

        logger.info(f"Loading HybridDeepfakeDetector from {resolved_path} onto {self.device}...")
        
        # Instantiate architecture
        model = HybridDeepfakeDetector(num_classes=settings.NUM_CLASSES)

        # Load weights
        checkpoint = torch.load(resolved_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        # Load state dict
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        self.model = model
        self.is_loaded = True
        logger.info(
            f"Successfully loaded model! Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}, "
            f"Validation Acc: {checkpoint.get('val_accuracy', 'N/A')}"
        )

        # Initialize prebuilt Vision Transformer (ViT-B/32) feature extractor
        try:
            logger.info("Initializing prebuilt Vision Transformer (ViT-B/32) for patch synthesis analysis...")
            vit = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
            vit.to(self.device)
            vit.eval()
            self.vit_model = vit
            self.vit_categories = models.ViT_B_32_Weights.DEFAULT.meta.get('categories', [])
            self.vit_transforms = models.ViT_B_32_Weights.DEFAULT.transforms()
            logger.info("Successfully loaded prebuilt ViT-B/32!")
        except Exception as ve:
            logger.warning(f"Could not load prebuilt ViT-B/32: {ve}. Proceeding with dual-branch hybrid model.")
            self.vit_model = None
            self.vit_categories = []
            self.vit_transforms = None

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocesses raw bytes into a multi-view Test-Time Augmentation (TTA) batch."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # EXIF Orientation correction: Prevents smartphone portrait photos from loading sideways
            image = ImageOps.exif_transpose(image)
            # Ensure 3-channel RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image format: {str(e)}")

        # Construct 3-view Test-Time Augmentation (TTA) batch:
        t_std = self.transform_std(image)
        t_flip = torch.flip(t_std, dims=[2])
        t_crop = self.transform_crop(image)

        # Stack into batch (3, 3, 224, 224)
        return torch.stack([t_std, t_flip, t_crop], dim=0).to(self.device)

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Runs image through hybrid model with TTA, frequency FFT, ViT scene context, and multi-signal disambiguation."""
        if not self.is_loaded or self.model is None:
            self.load_model()

        # 1. Structural & Metadata Analysis on original image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_rgb = pil_img.convert("RGB")
            w, h = pil_img.size
            
            # Check camera hardware EXIF tags (Make=271, Model=272, LensMake=42035)
            exif = pil_img.getexif()
            has_camera_hardware = bool(exif.get(271) or exif.get(272) or exif.get(42035))
        except Exception:
            pil_rgb = Image.new("RGB", (224, 224))
            w, h = 224, 224
            has_camera_hardware = False

        # 2. 2D Fast Fourier Transform (FFT) High-Frequency Spectral Energy Analysis
        arr = np.array(pil_rgb, dtype=np.float32) / 255.0
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        
        cy, cx = h // 2, w // 2
        r_outer = min(h, w) // 2
        r_inner = min(h, w) // 4
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
        hf_mask = (dist >= r_inner) & (dist <= r_outer)
        
        hf_energy = float(np.mean(mag[hf_mask]))
        total_energy = float(np.mean(mag))
        hf_ratio = round(hf_energy / (total_energy + 1e-6), 4)

        # Average gradient norm (measures micro-texture energy across the entire frame)
        dy, dx = np.gradient(gray * 255.0)
        grad_norm = round(float(np.mean(np.sqrt(dx ** 2 + dy ** 2))), 2)

        # 3. Error Level Analysis (ELA) - Localized Compression Discontinuity
        buf = io.BytesIO()
        pil_rgb.save(buf, "JPEG", quality=92)
        buf.seek(0)
        im_j = Image.open(buf)
        t_orig = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        t_comp = torch.tensor(np.array(im_j, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        diff = torch.abs(t_orig - t_comp)
        c_h, c_w = h // 4, w // 4
        mask_peri = torch.ones((1, 1, h, w))
        mask_peri[:, :, c_h:3*c_h, c_w:3*c_w] = 0.0
        ela_center = float(diff[:, :, c_h:3*c_h, c_w:3*c_w].mean().item())
        ela_peri = float((diff * mask_peri).sum().item() / (mask_peri.sum().item() * 3 + 1e-6))
        ela_ratio = round(ela_center / (ela_peri + 1e-6), 3)

        # 4. ViT-B/32 Scene Category & Patch Token Anomaly Analysis
        top_category = "general"
        is_landscape_or_scene = False
        vit_patch_anomaly = None
        vit_synthesis_score = None

        if self.vit_model is not None and self.vit_transforms is not None:
            try:
                inp_vit = self.vit_transforms(pil_rgb).unsqueeze(0).to(self.device)
                out_vit = self.vit_model(inp_vit)
                top1_idx = torch.argmax(out_vit, dim=1).item()
                if top1_idx < len(self.vit_categories):
                    top_category = self.vit_categories[top1_idx]
                
                is_landscape_or_scene = any(k in top_category.lower() for k in LANDSCAPE_SCENE_KEYWORDS)

                # Extract patch token anomaly across 49 patches
                v_in = inp_vit
                feat = self.vit_model.conv_proj(v_in).flatten(2).transpose(1, 2)
                b = v_in.shape[0]
                tokens = torch.cat([self.vit_model.class_token.expand(b, -1, -1), feat], dim=1) + self.vit_model.encoder.pos_embedding
                enc = self.vit_model.encoder.layers(self.vit_model.encoder.dropout(tokens))
                enc = self.vit_model.encoder.ln(enc)
                patch_tokens = enc[:, 1:]  # (1, 49, 768)
                patch_norms = torch.norm(patch_tokens, dim=-1)
                patch_std = float(patch_norms.std().item())
                vit_patch_anomaly = round(patch_std, 4)

                if patch_std < 0.65:
                    vit_synthesis_score = round(1.0 - (patch_std / 0.65) * 0.4, 3)
                else:
                    vit_synthesis_score = round(max(0.0, 0.6 - (patch_std - 0.65) * 0.5), 3)
            except Exception as e:
                logger.debug(f"ViT-B/32 analysis error: {e}")

        # 5. Dual-Branch PyTorch Neural Network Inference
        input_batch = self.preprocess(image_bytes)
        logits, forensics = self.model.forward_with_forensics(input_batch)
        mean_logits = torch.mean(logits, dim=0, keepdim=True).clone()

        real_idx = self.classes.index("real")
        ai_idx = self.classes.index("ai_generated")
        photo_idx = self.classes.index("photoshop")
        swap_idx = self.classes.index("faceswap")

        srm_energy = forensics.get("srm_residual_energy", 0.0)
        noise_inconsistency = forensics.get("noise_inconsistency", 0.0)

        # Check native AI resolution fingerprints (e.g. Gemini 1408x768 / 768x1408)
        is_ai_res = (w, h) in AI_GENERATIVE_RESOLUTIONS

        # 6. Targeted Multimodal Forensic Disambiguation:
        # Fuses the trained dual-branch model with ViT-B/32 patch metrics and ELA compression gradients
        # to correctly classify diverse real-world images (AI, FaceSwap, Photoshop, Real).
        raw_top_idx = int(torch.argmax(mean_logits, dim=1).item())
        raw_class = self.classes[raw_top_idx]

        # Rule 1: Spliced Photoshop detection (high localized noise inconsistency or high ELA on non-portrait)
        if (noise_inconsistency >= 0.45 and srm_energy >= 0.015 and not has_camera_hardware) or (ela_ratio > 3.0 and noise_inconsistency > 0.40):
            current_max = mean_logits[0].max().item()
            mean_logits[0, photo_idx] = max(current_max + 1.6, 2.5)
            mean_logits[0, real_idx] -= 2.0
            forensics["manipulation_profile"] = "Photoshop Digital Splicing / Localized Discontinuity"
            logger.info("Photoshop digital tampering detected via high localized noise inconsistency.")

        # Rule 2: FaceSwap localized facial graft (high ELA error in face area, but low general background inconsistency)
        elif (ela_ratio >= 3.0 and noise_inconsistency < 0.40 and not is_landscape_or_scene):
            current_max = mean_logits[0].max().item()
            mean_logits[0, swap_idx] = max(current_max + 1.6, 2.5)
            mean_logits[0, real_idx] -= 2.0
            forensics["manipulation_profile"] = "FaceSwap / Deepfake Boundary Graft"
            logger.info("FaceSwap facial boundary graft detected via ELA localized disparity.")

        # Rule 3: Generative AI full-scene synthesis (Diffusion / Midjourney / DALL-E / Gemini)
        elif raw_class in ("faceswap", "real") and not has_camera_hardware:
            if (is_ai_res and (srm_energy > 0.010 or hf_ratio > 0.85)) or \
               (is_landscape_or_scene and srm_energy > 0.015) or \
               (raw_class == "faceswap" and ela_ratio < 1.5 and vit_patch_anomaly is not None and vit_patch_anomaly > 0.95):
                current_max = mean_logits[0].max().item()
                mean_logits[0, ai_idx] = max(current_max + 1.6, 2.5)
                mean_logits[0, swap_idx] -= 2.0
                mean_logits[0, real_idx] -= 1.0
                forensics["generative_synthesis_confirmed"] = True
                forensics["ai_generator_profile"] = "Diffusion Latent Synthesis (Gemini/Imagen/Midjourney)"
                logger.info("Generative diffusion synthesis detected via full-canvas patch/spectral signature.")

        # Record augmented telemetry
        forensics["fft_high_frequency_ratio"] = hf_ratio
        forensics["gradient_micro_energy"] = grad_norm
        forensics["ela_disparity_ratio"] = ela_ratio
        forensics["vit_scene_category"] = top_category
        forensics["vit_32_patch_anomaly"] = vit_patch_anomaly
        forensics["vit_synthesis_score"] = vit_synthesis_score
        forensics["native_ai_resolution"] = is_ai_res
        forensics["camera_hardware_exif"] = has_camera_hardware


        # 6. Calibrated Softmax Confidence (T = 0.65)
        temperature = 0.65
        calibrated_logits = mean_logits / temperature
        probabilities = F.softmax(calibrated_logits, dim=1).squeeze(0).cpu().tolist()

        class_scores = []
        for idx, class_name in enumerate(self.classes):
            conf = float(probabilities[idx])
            class_scores.append({
                "class_name": class_name,
                "confidence": round(conf, 4),
                "percentage": round(conf * 100.0, 2)
            })

        sorted_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
        top_idx = sorted_indices[0]
        runner_up_idx = sorted_indices[1]

        predicted_class = self.classes[top_idx]
        runner_up_class = self.classes[runner_up_idx]
        top_confidence = round(float(probabilities[top_idx]), 4)
        runner_up_confidence = round(float(probabilities[runner_up_idx]), 4)
        margin = round(top_confidence - runner_up_confidence, 4)

        is_fake = predicted_class != "real"
        is_ambiguous = (margin < 0.20) and is_fake and (runner_up_class != "real")

        forensics["margin"] = margin
        forensics["runner_up_class"] = runner_up_class
        forensics["runner_up_confidence"] = runner_up_confidence
        forensics["is_ambiguous"] = is_ambiguous

        return {
            "predicted_class": predicted_class,
            "runner_up_class": runner_up_class,
            "confidence": top_confidence,
            "margin": margin,
            "is_fake": is_fake,
            "is_ambiguous": is_ambiguous,
            "class_probabilities": class_scores,
            "forensics": forensics,
        }


def get_model_service() -> ModelService:
    """Dependency injection helper."""
    return ModelService.get_instance()

