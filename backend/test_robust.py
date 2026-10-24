from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import io
from app.services.model_service import ModelService, LANDSCAPE_SCENE_KEYWORDS, AI_GENERATIVE_RESOLUTIONS

ms = ModelService.get_instance()
ms.load_model()

EDITING_SOFTWARE_KEYWORDS = ["photoshop", "adobe", "gimp", "picsart", "canva", "lightroom", "affinity"]
GENERATIVE_SOFTWARE_KEYWORDS = ["midjourney", "dall-e", "stable diffusion", "comfyui", "automatic1111", "novelai", "imagen"]

def robust_predict(image_bytes: bytes):
    pil_img = Image.open(io.BytesIO(image_bytes))
    w, h = pil_img.size
    exif = pil_img.getexif()
    has_cam = bool(exif.get(271) or exif.get(272) or exif.get(42035))
    software_str = str(exif.get(305, "")).lower()
    
    has_editing_software = any(k in software_str for k in EDITING_SOFTWARE_KEYWORDS)
    has_generative_software = any(k in software_str for k in GENERATIVE_SOFTWARE_KEYWORDS)

    # 1. Trained .pth model forward
    input_batch = ms.preprocess(image_bytes)
    logits, forensics = ms.model.forward_with_forensics(input_batch)
    mean_logits = torch.mean(logits, dim=0, keepdim=True).clone()

    # 2. 2D FFT & gradient
    pil_rgb = pil_img.convert("RGB")
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
    dy, dx = np.gradient(gray * 255.0)
    grad_norm = round(float(np.mean(np.sqrt(dx ** 2 + dy ** 2))), 2)

    # 3. ELA (Error Level Analysis) center vs periphery
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

    # 4. ViT
    top_cat = "general"
    is_landscape_or_scene = False
    vit_patch_anomaly = None
    if ms.vit_model is not None and ms.vit_transforms is not None:
        try:
            inp_vit = ms.vit_transforms(pil_rgb).unsqueeze(0).to(ms.device)
            out_vit = ms.vit_model(inp_vit)
            top1_idx = torch.argmax(out_vit, dim=1).item()
            if top1_idx < len(ms.vit_categories):
                top_cat = ms.vit_categories[top1_idx]
            is_landscape_or_scene = any(k in top_cat.lower() for k in LANDSCAPE_SCENE_KEYWORDS)
            v_in = inp_vit
            feat = ms.vit_model.conv_proj(v_in).flatten(2).transpose(1, 2)
            b = v_in.shape[0]
            tokens = torch.cat([ms.vit_model.class_token.expand(b, -1, -1), feat], dim=1) + ms.vit_model.encoder.pos_embedding
            enc = ms.vit_model.encoder.layers(ms.vit_model.encoder.dropout(tokens))
            enc = ms.vit_model.encoder.ln(enc)
            patch_tokens = enc[:, 1:]
            patch_norms = torch.norm(patch_tokens, dim=-1)
            vit_patch_anomaly = round(float(patch_norms.std().item()), 4)
        except Exception:
            pass

    srm_energy = forensics.get("srm_residual_energy", 0.0)
    noise_inconsistency = forensics.get("noise_inconsistency", 0.0)
    is_ai_res = (w, h) in AI_GENERATIVE_RESOLUTIONS

    real_idx = ms.classes.index("real")
    photo_idx = ms.classes.index("photoshop")
    ai_idx = ms.classes.index("ai_generated")
    swap_idx = ms.classes.index("faceswap")

    raw_top_idx = int(torch.argmax(mean_logits, dim=1).item())
    raw_class = ms.classes[raw_top_idx]

    # --- MULTI-SIGNAL FORENSIC DISAMBIGUATION ---
    # Case 1: EXIF Metadata Tampering Signatures
    if has_editing_software:
        current_max = mean_logits[0].max().item()
        mean_logits[0, photo_idx] = max(current_max + 1.8, 2.8)
        mean_logits[0, real_idx] -= 2.0
    elif has_generative_software:
        current_max = mean_logits[0].max().item()
        mean_logits[0, ai_idx] = max(current_max + 1.8, 2.8)
        mean_logits[0, real_idx] -= 2.0

    # Case 2: Spliced Photoshop Detection (high localized noise inconsistency or high ELA on non-portrait)
    elif (noise_inconsistency >= 0.45 and srm_energy >= 0.015) or (ela_ratio > 3.0 and noise_inconsistency > 0.40):
        current_max = mean_logits[0].max().item()
        mean_logits[0, photo_idx] = max(current_max + 1.6, 2.5)
        mean_logits[0, real_idx] -= 2.0

    # Case 3: FaceSwap Localized Facial Replacement (high ELA error in face area, but low general noise inconsistency)
    elif (ela_ratio >= 3.0 and noise_inconsistency < 0.40 and not is_landscape_or_scene):
        current_max = mean_logits[0].max().item()
        mean_logits[0, swap_idx] = max(current_max + 1.6, 2.5)
        mean_logits[0, real_idx] -= 2.0

    # Case 4: Generative AI full-scene synthesis (Diffusion / Midjourney / DALL-E / Gemini)
    # Catches when the model falsely predicts faceswap or defaults to real
    elif raw_class in ("faceswap", "real"):
        is_diffusion_canvas = (
            (is_ai_res and not has_cam and (srm_energy > 0.010 or hf_ratio > 0.85)) or
            (is_landscape_or_scene and not has_cam and srm_energy > 0.015) or
            (raw_class == "faceswap" and ela_ratio < 1.5 and vit_patch_anomaly is not None and vit_patch_anomaly > 0.95) or
            (hf_ratio > 0.94 and grad_norm > 10.0 and srm_energy > 0.018 and not has_cam)
        )
        if is_diffusion_canvas:
            current_max = mean_logits[0].max().item()
            mean_logits[0, ai_idx] = max(current_max + 1.6, 2.5)
            mean_logits[0, swap_idx] -= 2.0
            mean_logits[0, real_idx] -= 1.0

    probs = F.softmax(mean_logits / 0.65, dim=1).squeeze(0).tolist()
    top_idx = int(torch.argmax(mean_logits, dim=1).item())
    
    return ms.classes[top_idx], probs[top_idx]

print("TESTING 6ced6f98.jpg AND 9657bdfe.jpg (Photoshop images previously called Real):")
for name in ["6ced6f98.jpg", "9657bdfe.jpg"]:
    p = Path("app/uploads") / name
    if p.exists():
        c, conf = robust_predict(p.read_bytes())
        print(f"  {name}: {c} ({conf*100:.1f}%)")

print("\nTESTING 4 USER IMAGES:")
dir_path = Path(r"C:\Users\dell\.gemini\antigravity-ide\brain\c2e61f2f-0974-4252-97f7-4c3241d902c2\.user_uploaded")
for idx, p in enumerate(sorted(dir_path.glob("media_*.jpg"))):
    c, conf = robust_predict(p.read_bytes())
    print(f"  Image {idx+1} ({p.name}): {c} ({conf*100:.1f}%)")
