# === FILE: deploy/main.py ===
import ast
import inspect
import json
import re
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

cv2.setNumThreads(1)

# ==============================================================================
# ⚙️ CẤU HÌNH DEPLOY TRỰC TIẾP
# ==============================================================================
CFG = {
    "load_only_best": True,            # True: Chỉ nạp 3 mô hình *_best.pt
    "preferred_device": "XPU",         # "XPU", "CUDA", hoặc "CPU"
    "default_thresholds": {            # Ngưỡng mặc định dự phòng cho 3 model
        "model1_custom_cnn_best.pt": 0.3877,
        "model2_transfer_best.pt":   0.1987,
        "model3_multitask_best.pt":  0.1109,
    },
    "crop_margin_ratio": 0.35,         # Độ rộng viền cắt mặt chuẩn LCC-FASD (35%)
    "min_face_ratio": 0.12,            # Bắt khuôn mặt nhỏ khi ngồi xa màn hình
    "distance_sharpness_boost": 0.40,  # Bù nét khi mặt ở xa (< 175px), tự tắt khi ở gần
    "prob_smoothing_frames": 5,
    "camera_id": 0,
    "camera_width": 1280,
    "camera_height": 720,
    "show_trackbars": True,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE, MAP_SIZE = 192, 24


def disable_inplace_activations(module: nn.Module):
    for m in module.modules():
        if hasattr(m, "inplace") and isinstance(m.inplace, bool):
            m.inplace = False


# ==============================================================================
# 1. TRÍCH XUẤT ĐẶC TRƯNG STAGE 3 (YCrCb + 3D Depth + FFT/LBP Noise)
# ==============================================================================
def _build_fft_hp_mask(size: int = IMG_SIZE, cutoff: float = 0.12) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((y - size // 2) ** 2 + (x - size // 2) ** 2)
    return (1.0 - np.exp(-(dist ** 2) / (2.0 * (size * cutoff) ** 2 + 1e-6))).astype(np.float32)


def _build_3d_dome(size: int = IMG_SIZE) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.clip(1.0 - ((xx / 0.85) ** 2 + ((yy + 0.05) / 0.95) / 2), 0.0, 1.0) ** 0.65


FFT_HP_MASK = _build_fft_hp_mask(IMG_SIZE)
DOME_PRIOR  = _build_3d_dome(IMG_SIZE)


def extract_stage3_maps(face_rgb: np.ndarray, pred_depth: np.ndarray = None, pred_noise: np.ndarray = None):
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_RGB2BGR)

    if pred_depth is not None:
        d_map = cv2.resize(pred_depth.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        d_norm = np.clip((d_map - d_map.min()) / (d_map.max() - d_map.min() + 1e-6), 0.0, 1.0)
    else:
        ycrcb_f = ycrcb.astype(np.float32) / 255.0
        lum = cv2.GaussianBlur(ycrcb_f[:, :, 0], (9, 9), 0)
        cr  = cv2.GaussianBlur(ycrcb_f[:, :, 1], (9, 9), 0)
        d_est = (0.65 * DOME_PRIOR + 0.20 * lum + 0.15 * cr) * DOME_PRIOR
        d_norm = (d_est - d_est.min()) / (d_est.max() - d_est.min() + 1e-6)
    depth_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    f_high = np.fft.fftshift(np.fft.fft2(gray)) * FFT_HP_MASK
    img_hp = np.abs(np.fft.ifft2(np.fft.ifftshift(f_high)))

    pad = np.pad(gray, 1, mode="reflect")
    c = pad[1:-1, 1:-1]
    nbs = [pad[:-2, :-2], pad[:-2, 1:-1], pad[:-2, 2:], pad[1:-1, :-2], pad[1:-1, 2:], pad[2:, :-2], pad[2:, 1:-1], pad[2:, 2:]]
    lbp_diff = sum(np.abs(c - nb) for nb in nbs) / 8.0

    raw_noise = 0.50 * img_hp + 0.50 * lbp_diff
    p5, p98 = np.percentile(raw_noise, 5), np.percentile(raw_noise, 98)
    n_norm = np.clip((raw_noise - p5) / (p98 - p5 + 1e-6), 0.0, 1.0)

    if pred_noise is not None:
        pn = cv2.resize(np.clip(pred_noise.astype(np.float32), 0.0, 1.0), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        n_norm = 0.55 * n_norm + 0.45 * pn

    noise_color = cv2.applyColorMap((np.clip(n_norm, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return ycrcb_bgr, depth_color, noise_color


# ==============================================================================
# 2. KIẾN TRÚC MODEL 1, 2, 3 & TRÌNH ĐỒNG BỘ CLASS TỪ NOTEBOOK
# ==============================================================================
class Conv2d_CD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=0.4, direct_weight=False):
        super().__init__()
        self.theta, self.direct_weight, self.stride, self.padding = float(theta), direct_weight, stride, padding
        if direct_weight:
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
            self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight if self.direct_weight else self.conv.weight
        b = self.bias if self.direct_weight else self.conv.bias
        s = self.stride if self.direct_weight else self.conv.stride
        p = self.padding if self.direct_weight else self.conv.padding
        out = F.conv2d(x, w, bias=b, stride=s, padding=p)
        if abs(self.theta) < 1e-6:
            return out
        return out - self.theta * F.conv2d(x, w.sum(dim=(2, 3), keepdim=True), bias=None, stride=s, padding=0)


class DynamicSEBlock(nn.Module):
    def __init__(self, prefix: str, sd: dict):
        super().__init__()
        fc_keys = [k for k in sd if k.startswith(prefix + "fc.") and k.endswith(".weight")]
        idxs = sorted(int(k.split(".")[len(prefix.split("."))]) for k in fc_keys)
        self.use_conv = any(sd[k].ndim == 4 for k in fc_keys)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        layers = []
        if len(idxs) == 2:
            i0, i1 = idxs
            w0, w1 = sd[f"{prefix}fc.{i0}.weight"], sd[f"{prefix}fc.{i1}.weight"]
            b0, b1 = f"{prefix}fc.{i0}.bias" in sd, f"{prefix}fc.{i1}.bias" in sd
            for idx in range(i1 + 2):
                if idx == i0:
                    layers.append(nn.Conv2d(w0.shape[1], w0.shape[0], 1, bias=b0) if self.use_conv else nn.Linear(w0.shape[1], w0.shape[0], bias=b0))
                elif idx == i1:
                    layers.append(nn.Conv2d(w1.shape[1], w1.shape[0], 1, bias=b1) if self.use_conv else nn.Linear(w1.shape[1], w1.shape[0], bias=b1))
                elif idx < i0:
                    layers.append(nn.Identity())
                elif i0 < idx < i1:
                    layers.append(nn.SiLU(inplace=False))
                else:
                    layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            return x * self.fc(self.avg_pool(x))
        b, c, _, _ = x.size()
        return x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)


class DynamicResidualCDBlock(nn.Module):
    def __init__(self, prefix: str, sd: dict, theta: float = 0.4, stride: int = 2):
        super().__init__()
        dw = f"{prefix}conv1.weight" in sd
        w1 = sd[f"{prefix}conv1.weight" if dw else f"{prefix}conv1.conv.weight"]
        out_c, in_c = w1.shape[0], w1.shape[1]
        self.conv1 = Conv2d_CD(in_c, out_c, 3, stride, 1, False, theta, dw)
        self.bn1, self.act = nn.BatchNorm2d(out_c), nn.SiLU(inplace=False)
        self.conv2 = Conv2d_CD(out_c, out_c, 3, 1, 1, False, theta, dw)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = DynamicSEBlock(f"{prefix}se.", sd) if any(k.startswith(f"{prefix}se.") for k in sd) else nn.Identity()
        self.sc_name = next((c for c in ("shortcut", "downsample") if any(k.startswith(f"{prefix}{c}.") for k in sd)), None)
        if self.sc_name:
            sw = sd[f"{prefix}{self.sc_name}.0.weight"]
            sb = f"{prefix}{self.sc_name}.0.bias" in sd
            setattr(self, self.sc_name, nn.Sequential(nn.Conv2d(sw.shape[1], sw.shape[0], sw.shape[2], stride, bias=sb), nn.BatchNorm2d(sw.shape[0])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))))
        ident = getattr(self, self.sc_name)(x) if self.sc_name else x
        if ident.shape[2:] != out.shape[2:]:
            ident = F.adaptive_avg_pool2d(ident, out.shape[2:])
        return self.act(out + ident)


class CustomFASNet(nn.Module):
    def __init__(self, sd: dict, theta: float = 0.4):
        super().__init__()
        stem_w = [k for k in sd if k.startswith("stem.") and k.endswith("weight")]
        max_s = max((int(k.split(".")[1]) for k in stem_w), default=1) + 1
        stem = []
        for i in range(max_s + 1):
            if f"stem.{i}.conv.weight" in sd:
                w = sd[f"stem.{i}.conv.weight"]
                stem.append(Conv2d_CD(w.shape[1], w.shape[0], w.shape[2], 2, 1, theta=theta, direct_weight=False))
            elif f"stem.{i}.weight" in sd:
                w = sd[f"stem.{i}.weight"]
                stem.append(Conv2d_CD(w.shape[1], w.shape[0], w.shape[2], 2, 1, theta=theta, direct_weight=True) if w.ndim == 4 else nn.BatchNorm2d(w.shape[0]))
            else:
                stem.append(nn.SiLU(inplace=False))
        self.stem = nn.Sequential(*stem)

        self.stage_names = sorted(set(k.split(".")[0] for k in sd if re.match(r"^(stage|block|layer)\d+$", k.split(".")[0])))
        for s in self.stage_names:
            has_sc = any(k.startswith(f"{s}.shortcut.") or k.startswith(f"{s}.downsample.") for k in sd)
            setattr(self, s, DynamicResidualCDBlock(f"{s}.", sd, theta=theta, stride=2 if has_sc else 1))

        self.avg_pool, self.max_pool = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        cls_w = [k for k in sd if k.startswith("classifier.") and k.endswith(".weight")]
        max_c = max((int(k.split(".")[1]) for k in cls_w), default=0)
        cls = []
        for i in range(max_c + 1):
            wk, bk = f"classifier.{i}.weight", f"classifier.{i}.bias"
            if wk in sd:
                w = sd[wk]
                cls.append(nn.Linear(w.shape[1], w.shape[0], bias=(bk in sd)) if w.ndim == 2 else nn.BatchNorm1d(w.shape[0]))
            else:
                nxt_bn = f"classifier.{i+1}.weight" in sd and sd[f"classifier.{i+1}.weight"].ndim == 1
                cls.append(nn.Identity() if nxt_bn else nn.SiLU(inplace=False))
        self.classifier = nn.Sequential(*cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for s in self.stage_names:
            x = getattr(self, s)(x)
        return self.classifier(torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1))


class TransferFASNet(nn.Module):
    def __init__(self, backbone_name: str = "mobilenet_v3_large", num_classes: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        bname = backbone_name.lower()
        if bname == "mobilenet_v3_large":
            self.encoder, self.out_channels = tv_models.mobilenet_v3_large(weights=None).features, 960
        elif bname == "efficientnet_b0":
            self.encoder, self.out_channels = tv_models.efficientnet_b0(weights=None).features, 1280
        elif bname == "resnet34":
            b = tv_models.resnet34(weights=None)
            self.encoder = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool, b.layer1, b.layer2, b.layer3, b.layer4)
            self.out_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        disable_inplace_activations(self.encoder)
        self.avg_pool, self.max_pool = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels * 2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=False),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        return self.classifier(torch.cat([self.avg_pool(f).flatten(1), self.max_pool(f).flatten(1)], dim=1))


def load_notebook_model_classes(project_root: Path) -> list:
    env = {
        "torch": torch, "nn": nn, "F": F, "tv_models": tv_models,
        "np": np, "cv2": cv2, "Path": Path, "re": re,
        "IMG_SIZE": IMG_SIZE, "MAP_SIZE": MAP_SIZE, "NUM_CLASSES": 2,
        "BACKBONE_NAME": "mobilenet_v3_large", "THETA_CD": 0.4,
        "MODEL2_BEST_CKPT_PATH": project_root / "models" / "model2_transfer_best.pt",
        "DEVICE": torch.device("cpu"), "has_xpu": False, "USE_AMP": False,
        "AUTOCAST_DEVICE_TYPE": "cpu", "AUTOCAST_DTYPE": torch.float32,
        "disable_inplace_activations": disable_inplace_activations,
        "Conv2d_CD": Conv2d_CD,
    }
    found_classes = []
    for nb_path in sorted(project_root.rglob("*.ipynb")):
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            for cell in nb.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                src = "".join(cell.get("source", []))
                if "class " not in src or "nn.Module" not in src:
                    continue
                clean_lines = [ln for ln in src.splitlines() if not ln.strip().startswith(("%", "!"))]
                tree = ast.parse("\n".join(clean_lines))
                defs = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
                exec(compile(ast.Module(body=defs, type_ignores=[]), str(nb_path.name), "exec"), env)
        except Exception:
            pass

    for val in env.values():
        if isinstance(val, type) and issubclass(val, nn.Module) and val is not nn.Module:
            if val not in (Conv2d_CD, DynamicSEBlock, DynamicResidualCDBlock):
                found_classes.append(val)
    return found_classes


# ==============================================================================
# 3. NẠP 3 MÔ HÌNH *_best.pt VÀ NGƯỠNG QUYẾT ĐỊNH TỐI ƯU
# ==============================================================================
def get_compute_device(preferred: str = "XPU"):
    pref = preferred.upper()
    if pref in ("XPU", "AUTO") and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0"), True, torch.bfloat16
    if pref in ("CUDA", "AUTO") and torch.cuda.is_available():
        return torch.device("cuda:0"), True, torch.float16
    return torch.device("cpu"), False, torch.float32


def instantiate_model_from_ckpt(ckpt_path: Path, ckpt: dict, state_dict: dict, nb_classes: list):
    m_cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    b_params = ckpt.get("best_params", {}) if isinstance(ckpt, dict) else {}
    merged_cfg = {**b_params, **m_cfg}
    fname = ckpt_path.name.lower()

    for cls in reversed(nb_classes):
        try:
            sig = inspect.signature(cls.__init__)
            kwargs = {k: v for k, v in merged_cfg.items() if k in sig.parameters}
            if "pretrained" in sig.parameters:
                kwargs["pretrained"] = False
            if "load_stage2_weights" in sig.parameters:
                kwargs["load_stage2_weights"] = False
            model = cls(**kwargs)
            model.load_state_dict(state_dict, strict=True)
            return model
        except Exception:
            continue

    bname = ckpt.get("backbone_name", merged_cfg.get("backbone_name", None)) if isinstance(ckpt, dict) else None
    if "model2" in fname or (bname and set(k.split(".")[0] for k in state_dict) <= {"encoder", "classifier"}):
        dropout = float(merged_cfg.get("dropout_rate", 0.4))
        model = TransferFASNet(backbone_name=bname or "mobilenet_v3_large", num_classes=2, dropout_rate=dropout)
        model.load_state_dict(state_dict, strict=True)
        return model

    theta = float(merged_cfg.get("theta_cd", merged_cfg.get("theta", 0.4)))
    model = CustomFASNet(sd=state_dict, theta=theta)
    model.load_state_dict(state_dict, strict=True)
    return model


def load_models(models_dir: Path, project_root: Path, device: torch.device, use_channels_last: bool):
    pattern = "*_best.pt" if CFG["load_only_best"] else "*.pt"
    ckpt_files = sorted(models_dir.glob(pattern))
    if not ckpt_files:
        raise FileNotFoundError(f"❌ Không tìm thấy checkpoint ({pattern}) tại {models_dir}")

    nb_classes = load_notebook_model_classes(project_root)
    loaded = []

    for ckpt_path in ckpt_files:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            m_cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
            b_params = ckpt.get("best_params", {}) if isinstance(ckpt, dict) else {}

            fallback_thr = CFG["default_thresholds"].get(ckpt_path.name, 0.25)
            raw_thr = float(ckpt.get("val_threshold", fallback_thr)) if isinstance(ckpt, dict) else fallback_thr
            img_size = int(m_cfg.get("img_size", b_params.get("img_size", IMG_SIZE)))

            best_acer = ckpt.get("best_val_acer", None) if isinstance(ckpt, dict) else None
            if best_acer is not None and float(best_acer) < 1.0:
                best_acer = float(best_acer) * 100.0

            model = instantiate_model_from_ckpt(ckpt_path, ckpt, state_dict, nb_classes)
            disable_inplace_activations(model)
            model.eval()

            is_stage3 = "model3" in ckpt_path.name.lower() or "multitask" in ckpt_path.name.lower()
            is_stage2 = "model2" in ckpt_path.name.lower() or "transfer" in ckpt_path.name.lower()
            stage_label = "Model 3 (MultiTask: YCrCb+Depth+Noise)" if is_stage3 else (
                "Model 2 (Transfer: MobileNetV3-L)" if is_stage2 else "Model 1 (Scratch: CustomFASNet)"
            )

            aux_buffers = {"depth": None, "noise": None}
            if is_stage3:
                for name, module in model.named_modules():
                    lname = name.lower()
                    if any(k in lname for k in ("depth_head", "depth_decoder", "depth_out")):
                        module.register_forward_hook(lambda m, inp, out: aux_buffers.__setitem__("depth", out))
                    elif any(k in lname for k in ("noise_head", "noise_decoder", "noise_out")):
                        module.register_forward_hook(lambda m, inp, out: aux_buffers.__setitem__("noise", out))

            with torch.no_grad():
                _ = model(torch.zeros(1, 3, img_size, img_size))

            model = model.to(device, memory_format=torch.channels_last) if use_channels_last else model.to(device)
            loaded.append({
                "file_name": ckpt_path.name,
                "display_name": stage_label,
                "is_stage3": is_stage3,
                "model": model,
                "aux_buffers": aux_buffers,
                "threshold": raw_thr,
                "default_threshold": raw_thr,
                "img_size": img_size,
                "best_val_acer": best_acer,
            })
            acer_txt = f"{best_acer:.2f}%" if best_acer is not None else "N/A"
            print(f"✅ [{len(loaded)}] {ckpt_path.name:26s} | {stage_label:38s} | τ* = {raw_thr:.4f} | Val ACER = {acer_txt}")
        except Exception as e:
            print(f"⚠️ Bỏ qua '{ckpt_path.name}': {e}")

    if not loaded:
        raise RuntimeError("❌ Không nạp được mô hình nào!")
    return loaded


# ==============================================================================
# 4. BỘ DÒ MẶT YUNET DNN (TƯƠNG THÍCH OPENCV 5.0)
# ==============================================================================
class FaceDetectorLCC:
    YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    def __init__(self, margin_ratio: float = 0.35, min_face_ratio: float = 0.12):
        self.margin_ratio = margin_ratio
        self.min_face_ratio = min_face_ratio
        self.smooth_box = None
        self.missed = 0
        self.yunet = self._init_yunet()
        self.haar = self._init_haar_fallback()

    def _init_yunet(self):
        if not hasattr(cv2, "FaceDetectorYN"):
            return None
        onnx_path = Path(__file__).resolve().parent / "face_detection_yunet_2023mar.onnx"
        if not onnx_path.exists():
            try:
                import urllib.request
                urllib.request.urlretrieve(self.YUNET_URL, str(onnx_path))
            except Exception:
                return None
        try:
            return cv2.FaceDetectorYN.create(str(onnx_path), "", (320, 320), 0.60, 0.30, 5000)
        except Exception:
            return None

    @staticmethod
    def _init_haar_fallback():
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
            xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if Path(xml_path).exists():
                return cv2.CascadeClassifier(xml_path)
        return None

    @staticmethod
    def enhance_distant_face(face_rgb: np.ndarray, raw_crop_size: int, boost_strength: float) -> np.ndarray:
        if boost_strength <= 0.01 or raw_crop_size >= 175:
            return face_rgb
        scale_deficit = min(1.0, (175.0 - raw_crop_size) / 95.0)
        amount = boost_strength * scale_deficit
        blurred = cv2.GaussianBlur(face_rgb, (0, 0), sigmaX=1.1)
        return np.clip(cv2.addWeighted(face_rgb, 1.0 + amount, blurred, -amount, 0), 0, 255).astype(np.uint8)

    def detect_and_crop(self, frame_bgr: np.ndarray, target_size: int = 192, sharpness_boost: float = 0.4):
        h_img, w_img = frame_bgr.shape[:2]
        min_side = max(55, int(min(h_img, w_img) * self.min_face_ratio))
        raw_box = None

        if self.yunet is not None:
            scale = 640.0 / w_img
            det_h = int(round(h_img * scale))
            self.yunet.setInputSize((640, det_h))
            _, faces = self.yunet.detect(cv2.resize(frame_bgr, (640, det_h)))
            if faces is not None and len(faces) > 0:
                valid = [(int(f[0]/scale), int(f[1]/scale), int(f[2]/scale), int(f[3]/scale)) for f in faces if min(f[2]/scale, f[3]/scale) >= min_side]
                if valid:
                    raw_box = max(valid, key=lambda b: b[2] * b[3])

        if raw_box is None and self.haar is not None:
            gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
            faces = self.haar.detectMultiScale(gray, 1.1, 4, minSize=(min_side, min_side))
            if len(faces) > 0:
                raw_box = max(faces, key=lambda b: b[2] * b[3])

        if raw_box is not None:
            self.missed = 0
            x, y, w, h = raw_box
            pw, pt, pb = int(w * self.margin_ratio), int(h * self.margin_ratio * 1.2), int(h * self.margin_ratio * 0.85)
            cbox = np.array([max(0, x - pw), max(0, y - pt), min(w_img, x + w + pw), min(h_img, y + h + pb)], dtype=np.float32)
            self.smooth_box = cbox if self.smooth_box is None else (0.6 * self.smooth_box + 0.4 * cbox)
        else:
            self.missed += 1
            if self.smooth_box is None or self.missed > 8:
                self.smooth_box = None
                return None, None, 0

        x1, y1, x2, y2 = self.smooth_box.astype(int)
        crop_bgr = frame_bgr[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
        if crop_bgr.size == 0:
            return None, None, 0

        raw_px = min(x2 - x1, y2 - y1)
        face_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (target_size, target_size), interpolation=cv2.INTER_CUBIC if raw_px < target_size else cv2.INTER_AREA)
        face_rgb = self.enhance_distant_face(face_rgb, raw_px, sharpness_boost)

        return face_rgb, (x1, y1, x2, y2), raw_px


@torch.no_grad()
def run_inference(model_info: dict, face_rgb: np.ndarray, device: torch.device, use_amp: bool, amp_dtype: torch.dtype, channels_last: bool):
    model_info["aux_buffers"]["depth"] = None
    model_info["aux_buffers"]["noise"] = None

    img_norm = (face_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.ascontiguousarray(img_norm.transpose(2, 0, 1)[None, ...], dtype=np.float32))
    tensor = tensor.to(device, non_blocking=True, memory_format=torch.channels_last) if channels_last else tensor.to(device, non_blocking=True)

    model = model_info["model"]
    sig_fwd = inspect.signature(model.forward)
    fwd_kwargs = {k: True for k in ("return_aux", "return_maps", "return_all") if k in sig_fwd.parameters}

    dev_type = "xpu" if device.type == "xpu" else ("cuda" if device.type == "cuda" else "cpu")
    with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
        out = model(tensor, **fwd_kwargs)

    pred_depth, pred_noise = None, None
    if isinstance(out, (tuple, list)):
        logits = out[0]
        if len(out) > 1 and isinstance(out[1], torch.Tensor):
            pred_depth = out[1].float().cpu().numpy().squeeze()
        if len(out) > 2 and isinstance(out[2], torch.Tensor):
            pred_noise = out[2].float().cpu().numpy().squeeze()
    elif isinstance(out, dict):
        logits = out.get("logits", out.get("cls", out.get("out")))
        if "depth" in out and isinstance(out["depth"], torch.Tensor):
            pred_depth = out["depth"].float().cpu().numpy().squeeze()
        if "noise" in out and isinstance(out["noise"], torch.Tensor):
            pred_noise = out["noise"].float().cpu().numpy().squeeze()
    else:
        logits = out

    if pred_depth is None and isinstance(model_info["aux_buffers"]["depth"], torch.Tensor):
        pred_depth = model_info["aux_buffers"]["depth"].float().cpu().numpy().squeeze()
    if pred_noise is None and isinstance(model_info["aux_buffers"]["noise"], torch.Tensor):
        pred_noise = model_info["aux_buffers"]["noise"].float().cpu().numpy().squeeze()

    prob_real = float(F.softmax(logits.float(), dim=-1)[0, 1].item())
    return prob_real, pred_depth, pred_noise


def draw_diagnostic_panel(frame: np.ndarray, face_rgb: np.ndarray, is_stage3: bool, border_color: tuple, pred_depth=None, pred_noise=None):
    h_disp, w_disp = frame.shape[:2]
    rgb_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

    if not is_stage3:
        sz = 150
        thumb = cv2.resize(rgb_bgr, (sz, sz), interpolation=cv2.INTER_AREA)
        x1, y1 = w_disp - sz - 16, h_disp - sz - 16
        frame[y1:y1 + sz, x1:x1 + sz] = thumb
        cv2.rectangle(frame, (x1, y1), (x1 + sz, y1 + sz), border_color, 2)
        cv2.putText(frame, "RGB Crop", (x1 + 6, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return

    ycrcb_bgr, depth_bgr, noise_bgr = extract_stage3_maps(face_rgb, pred_depth, pred_noise)
    tiles = [
        ("1. RGB Crop", rgb_bgr),
        ("2. YCrCb Space", ycrcb_bgr),
        ("3. 3D Depth Map", depth_bgr),
        ("4. FFT+LBP Noise", noise_bgr),
    ]
    sz, gap = 128, 8
    panel_w = sz * 2 + gap * 3
    panel_h = sz * 2 + gap * 3 + 24
    px1, py1 = w_disp - panel_w - 14, h_disp - panel_h - 14

    sub = frame[py1:py1 + panel_h, px1:px1 + panel_w]
    dark = np.full_like(sub, 20)
    cv2.addWeighted(dark, 0.75, sub, 0.25, 0, sub)
    cv2.rectangle(frame, (px1, py1), (px1 + panel_w, py1 + panel_h), border_color, 2)
    cv2.putText(frame, "STAGE 3 MULTI-TASK MAPS", (px1 + 12, py1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 255), 1, cv2.LINE_AA)

    for idx, (title, img) in enumerate(tiles):
        r, c = divmod(idx, 2)
        tx = px1 + gap + c * (sz + gap)
        ty = py1 + 26 + r * (sz + gap)
        t_img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_AREA)
        frame[ty:ty + sz, tx:tx + sz] = t_img
        cv2.rectangle(frame, (tx, ty), (tx + sz, ty + sz), (180, 180, 180), 1)
        cv2.rectangle(frame, (tx, ty), (tx + sz, ty + 20), (0, 0, 0), -1)
        cv2.putText(frame, title, (tx + 5, ty + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


# ==============================================================================
# 5. BỘ HIỂN THỊ ĐA NỀN TẢNG (TỰ ĐỘNG FALLBACK SANG TKINTER NẾU OPENCV HEADLESS)
# ==============================================================================
class SmartDisplayWindow:
    def __init__(self, title: str, width: int, height: int, on_thr, on_margin, on_sharp, init_thr: int, init_margin: int, init_sharp: int):
        self.title = title
        self.width, self.height = width, height
        self.on_thr, self.on_margin, self.on_sharp = on_thr, on_margin, on_sharp
        self.use_tk = False
        self.last_key = -1
        self.closed = False
        self.is_fullscreen = False

        try:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, width, height)
            if CFG["show_trackbars"]:
                cv2.createTrackbar("Threshold (%)", self.title, init_thr, 95, self.on_thr)
                cv2.createTrackbar("Crop Margin (%)", self.title, init_margin, 60, self.on_margin)
                cv2.createTrackbar("Dist Sharpness", self.title, init_sharp, 150, self.on_sharp)
        except cv2.error:
            print("ℹ️ Phát hiện OpenCV Headless -> Tự động chuyển sang giao diện Tkinter GUI!")
            self.use_tk = True
            self._init_tkinter(init_thr, init_margin, init_sharp)

    def _init_tkinter(self, init_thr: int, init_margin: int, init_sharp: int):
        import tkinter as tk
        from PIL import Image, ImageTk
        self._tk = tk
        self._Image = Image
        self._ImageTk = ImageTk

        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.configure(bg="#181818")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Key>", self._on_key)

        if CFG["show_trackbars"]:
            ctrl = tk.Frame(self.root, bg="#222222", pady=4)
            ctrl.pack(fill=tk.X, side=tk.TOP)

            self.s_thr = self._make_slider(ctrl, "Threshold (%)", 1, 95, init_thr, lambda v: self.on_thr(int(float(v))))
            self.s_mar = self._make_slider(ctrl, "Crop Margin (%)", 5, 60, init_margin, lambda v: self.on_margin(int(float(v))))
            self.s_shp = self._make_slider(ctrl, "Dist Sharpness", 0, 150, init_sharp, lambda v: self.on_sharp(int(float(v))))

        self.video_label = tk.Label(self.root, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def _make_slider(self, parent, label, mn, mx, val, cmd):
        tk = self._tk
        frame = tk.Frame(parent, bg="#222222", padx=10)
        frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(frame, text=label, fg="#dcdcdc", bg="#222222", font=("Sans", 9, "bold")).pack(anchor="w")
        s = tk.Scale(frame, from_=mn, to=mx, orient=tk.HORIZONTAL, bg="#222222", fg="#50ff78",
                     highlightthickness=0, troughcolor="#383838", command=cmd)
        s.set(val)
        s.pack(fill=tk.X)
        return s

    def _on_close(self):
        self.closed = True

    def _on_key(self, event):
        ks = event.keysym
        if ks == "Escape":
            self.last_key = 27
        elif ks == "Tab":
            self.last_key = 9
        elif len( event.char ) == 1:
            self.last_key = ord(event.char)

    def sync_sliders(self, thr: int, margin: int, sharp: int):
        if not CFG["show_trackbars"]:
            return
        if not self.use_tk:
            cv2.setTrackbarPos("Threshold (%)", self.title, thr)
            cv2.setTrackbarPos("Crop Margin (%)", self.title, margin)
            cv2.setTrackbarPos("Dist Sharpness", self.title, sharp)
        else:
            self.s_thr.set(thr)
            self.s_mar.set(margin)
            self.s_shp.set(sharp)

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        if not self.use_tk:
            cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.is_fullscreen else cv2.WINDOW_NORMAL)
        else:
            self.root.attributes("-fullscreen", self.is_fullscreen)

    def show_and_wait_key(self, frame_bgr: np.ndarray) -> int:
        if not self.use_tk:
            cv2.imshow(self.title, frame_bgr)
            return cv2.waitKey(1) & 0xFF
        else:
            if self.closed:
                return ord("q")
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_tk = self._ImageTk.PhotoImage(image=self._Image.fromarray(rgb))
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk
            self.root.update_idletasks()
            self.root.update()
            k = self.last_key
            self.last_key = -1
            return k

    def destroy(self):
        if not self.use_tk:
            cv2.destroyAllWindows()
        elif not self.closed:
            self.root.destroy()


# ==============================================================================
# 6. VÒNG LẶP CHÍNH LIVE CAMERA
# ==============================================================================
def main():
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models" if (project_root / "models").exists() else Path("models").resolve()

    device, use_amp, amp_dtype = get_compute_device(CFG["preferred_device"])
    channels_last = (device.type == "xpu")

    print("=" * 95)
    print(f"🚀 FACE ANTI-SPOOFING DEPLOYMENT (3-MODEL SWITCHER) | Thiết bị: {device}")
    print("=" * 95)

    models_list = load_models(models_dir, project_root, device, channels_last)
    active_idx = len(models_list) - 1

    detector = FaceDetectorLCC(CFG["crop_margin_ratio"], CFG["min_face_ratio"])
    prob_history = deque(maxlen=CFG["prob_smoothing_frames"])
    sharpness_boost = float(CFG["distance_sharpness_boost"])
    show_preview = True

    cap = cv2.VideoCapture(CFG["camera_id"])
    if not cap.isOpened():
        raise RuntimeError("❌ Không thể mở Webcam!")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["camera_height"])

    def on_thr_change(val):
        models_list[active_idx]["threshold"] = max(0.01, min(0.99, val / 100.0))

    def on_margin_change(val):
        detector.margin_ratio = max(0.05, min(0.60, val / 100.0))

    def on_sharp_change(val):
        nonlocal sharpness_boost
        sharpness_boost = val / 100.0

    gui = SmartDisplayWindow(
        title="Face Anti-Spoofing Live (Press 1, 2, 3 or TAB to switch models)",
        width=CFG["camera_width"], height=CFG["camera_height"],
        on_thr=on_thr_change, on_margin=on_margin_change, on_sharp=on_sharp_change,
        init_thr=int(round(models_list[active_idx]["threshold"] * 100)),
        init_margin=int(round(detector.margin_ratio * 100)),
        init_sharp=int(round(sharpness_boost * 100)),
    )

    def sync_trackbars():
        gui.sync_sliders(
            int(round(models_list[active_idx]["threshold"] * 100)),
            int(round(detector.margin_ratio * 100)),
            int(round(sharpness_boost * 100)),
        )

    prev_time, fps_smooth = time.time(), 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h_disp, w_disp = frame.shape[:2]
        curr = models_list[active_idx]
        thr = float(curr["threshold"])

        t0 = time.time()
        face_rgb, box, raw_px = detector.detect_and_crop(frame, curr["img_size"], sharpness_boost)

        if face_rgb is not None:
            prob_real, pred_d, pred_n = run_inference(curr, face_rgb, device, use_amp, amp_dtype, channels_last)
            prob_history.append(prob_real)
            smooth_p = float(np.mean(prob_history))
            infer_ms = (time.time() - t0) * 1000.0

            is_real = smooth_p >= thr
            color = (0, 215, 0) if is_real else (0, 0, 235)
            status = "REAL (NGUOI THAT)" if is_real else "SPOOF (GIA MAO)"
            dist_tag = f"Gan ({raw_px}px - Goc)" if raw_px >= 175 else f"Xa ({raw_px}px - Boost)"

            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"{status} | P(Real)={smooth_p:.2f} (Thr={thr:.2f}) | {dist_tag}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 14)), (x1 + tw + 14, y1), color, -1)
            cv2.putText(frame, label, (x1 + 7, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

            if show_preview:
                draw_diagnostic_panel(frame, face_rgb, curr["is_stage3"], color, pred_d, pred_n)
        else:
            prob_history.clear()
            infer_ms = 0.0
            cv2.putText(frame, "Khong phat hien khuon mat", (20, h_disp - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 215, 255), 2, cv2.LINE_AA)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-5)
        prev_time = now
        fps_smooth = 0.85 * fps_smooth + 0.15 * fps if fps_smooth > 0 else fps

        hud = frame[0:80, 0:w_disp]
        cv2.addWeighted(np.full_like(hud, 20), 0.75, hud, 0.25, 0, hud)
        line1 = f"[{active_idx + 1}/{len(models_list)}] {curr['display_name']} ({curr['file_name']}) | FPS: {fps_smooth:.1f} ({infer_ms:.1f}ms)"
        line2 = (
            f"[1..{len(models_list)}/TAB]: Model | [[/]]: Thr={thr:.2f} | "
            f"[-/=]: Margin={int(detector.margin_ratio*100)}% | [,/.]: FarBoost={sharpness_boost:.2f} | [R]: Reset | [P]: Maps"
        )
        cv2.putText(frame, line1, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(frame, line2, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 225, 225), 1, cv2.LINE_AA)

        key = gui.show_and_wait_key(frame)

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (9, ord("m"), ord("M")):
            active_idx = (active_idx + 1) % len(models_list)
            prob_history.clear()
            sync_trackbars()
        elif ord("1") <= key <= ord("9") and (key - ord("1")) < len(models_list):
            active_idx = key - ord("1")
            prob_history.clear()
            sync_trackbars()
        elif key == ord("["):
            curr["threshold"] = max(0.01, curr["threshold"] - 0.02)
            sync_trackbars()
        elif key == ord("]"):
            curr["threshold"] = min(0.95, curr["threshold"] + 0.02)
            sync_trackbars()
        elif key in (ord("-"), ord("_")):
            detector.margin_ratio = max(0.05, detector.margin_ratio - 0.02)
            sync_trackbars()
        elif key in (ord("="), ord("+")):
            detector.margin_ratio = min(0.60, detector.margin_ratio + 0.02)
            sync_trackbars()
        elif key in (ord(","), ord("<")):
            sharpness_boost = max(0.0, sharpness_boost - 0.05)
            sync_trackbars()
        elif key in (ord("."), ord(">")):
            sharpness_boost = min(1.50, sharpness_boost + 0.05)
            sync_trackbars()
        elif key in (ord("r"), ord("R")):
            curr["threshold"] = curr["default_threshold"]
            detector.margin_ratio = CFG["crop_margin_ratio"]
            sharpness_boost = CFG["distance_sharpness_boost"]
            sync_trackbars()
        elif key in (ord("p"), ord("P")):
            show_preview = not show_preview
        elif key in (ord("f"), ord("F")):
            gui.toggle_fullscreen()

    cap.release()
    gui.destroy()


if __name__ == "__main__":
    main()