# === FILE: deploy/main.py ===
import os
import re
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

cv2.setNumThreads(1)

# ==============================================================================
# ⚙️ BẢNG CẤU HÌNH NGƯỜI DÙNG (CHỈNH SỬA TRỰC TIẾP TẠI ĐÂY)
# ==============================================================================
USER_CONFIG = {
    # 1. Nạp cả best + latest (False) hay chỉ nạp các file *_best.pt (True)?
    "load_only_best": False,

    # 2. Thiết bị chạy ("XPU" hoặc "CPU")
    "preferred_device": "XPU",

    # 3. Ngưỡng quyết định P(Real) >= threshold -> REAL cho từng model
    "override_default_threshold": None,
    "custom_model_thresholds": {
        "model1_custom_cnn_best.pt":   0.38,
        "model2_transfer_best.pt":     0.20,
    },

    # 4. Cấu hình Dò mặt & Bù nét (CHỈ bù nét khi mặt ở xa < 175px, tắt hẳn khi ở gần)
    "crop_margin_ratio": 0.35,       # Độ rộng viền cắt mặt chuẩn LCC-FASD (35%)
    "min_face_ratio": 0.12,          # Bắt được khuôn mặt nhỏ khi ngồi xa camera
    "distance_sharpness_boost": 0.40,# Mặc định nhẹ (0.40) và TỰ ĐỘNG = 0 khi ở gần camera
    "prob_smoothing_frames": 5,

    # 5. Giao diện & Camera
    "camera_id": 0,
    "camera_width": 1280,
    "camera_height": 720,
    "show_trackbars": True,
    "show_crop_preview": True,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==============================================================================
# 1. KIẾN TRÚC MODEL 1 (TỰ ĐỘNG KHỚP 100% THEO STATE_DICT CỦA STAGE 1)
# ==============================================================================
class Conv2d_CD(nn.Module):
    """Tích chập Sai phân Trung tâm (Central Difference Convolution) từ Stage 1."""
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, bias=False, theta=0.4, direct_weight=False
    ):
        super().__init__()
        self.theta = float(theta)
        self.direct_weight = direct_weight
        self.stride = stride
        self.padding = padding
        if direct_weight:
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_channels))
            else:
                self.register_parameter("bias", None)
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight if self.direct_weight else self.conv.weight
        b = self.bias if self.direct_weight else self.conv.bias
        s = self.stride if self.direct_weight else self.conv.stride
        p = self.padding if self.direct_weight else self.conv.padding

        out_normal = F.conv2d(x, w, bias=b, stride=s, padding=p)
        if abs(self.theta) < 1e-6:
            return out_normal
        kernel_diff = w.sum(dim=(2, 3), keepdim=True)
        out_diff = F.conv2d(x, kernel_diff, bias=None, stride=s, padding=0)
        return out_normal - self.theta * out_diff


class DynamicSEBlock(nn.Module):
    """Tự động dựng đúng cấu trúc SEBlock từ state_dict (Đã vá lỗi Pooling trùng lặp)."""
    def __init__(self, prefix: str, state_dict: dict):
        super().__init__()
        self.fc = nn.Sequential()
        fc_keys = [k for k in state_dict.keys() if k.startswith(prefix + "fc.") and k.endswith(".weight")]
        fc_indices = sorted(int(k.split(".")[len(prefix.split("."))]) for k in fc_keys)

        self.use_conv = any(state_dict[k].ndim == 4 for k in fc_keys)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if len(fc_indices) == 2:
            i0, i1 = fc_indices
            w0, w1 = state_dict[f"{prefix}fc.{i0}.weight"], state_dict[f"{prefix}fc.{i1}.weight"]
            b0 = f"{prefix}fc.{i0}.bias" in state_dict
            b1 = f"{prefix}fc.{i1}.bias" in state_dict
            max_idx = i1 + 1
            layers = []
            for idx in range(max_idx + 1):
                if idx == i0:
                    layers.append(
                        nn.Conv2d(w0.shape[1], w0.shape[0], 1, bias=b0)
                        if self.use_conv else nn.Linear(w0.shape[1], w0.shape[0], bias=b0)
                    )
                elif idx == i1:
                    layers.append(
                        nn.Conv2d(w1.shape[1], w1.shape[0], 1, bias=b1)
                        if self.use_conv else nn.Linear(w1.shape[1], w1.shape[0], bias=b1)
                    )
                elif idx < i0:
                    # Dùng Identity vì đã thực hiện avg_pool và view(b, c) ở đầu hàm forward
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
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)


class DynamicResidualCDBlock(nn.Module):
    """Tự động dựng khối ResidualCDBlock khớp tuyệt đối với state_dict."""
    def __init__(self, prefix: str, state_dict: dict, theta: float = 0.4, stride: int = 2):
        super().__init__()
        direct_w = f"{prefix}conv1.weight" in state_dict
        w1_key = f"{prefix}conv1.weight" if direct_w else f"{prefix}conv1.conv.weight"
        w2_key = f"{prefix}conv2.weight" if direct_w else f"{prefix}conv2.conv.weight"

        w1 = state_dict[w1_key]
        out_c, in_c = w1.shape[0], w1.shape[1]

        sc_name = None
        for cand in ("shortcut", "downsample"):
            if any(k.startswith(f"{prefix}{cand}.") for k in state_dict.keys()):
                sc_name = cand
                break

        self.conv1 = Conv2d_CD(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False, theta=theta, direct_weight=direct_w)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.act   = nn.SiLU(inplace=False)
        self.conv2 = Conv2d_CD(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False, theta=theta, direct_weight=direct_w)
        self.bn2   = nn.BatchNorm2d(out_c)

        if any(k.startswith(f"{prefix}se.") for k in state_dict.keys()):
            self.se = DynamicSEBlock(f"{prefix}se.", state_dict)
        else:
            self.se = nn.Identity()

        self.sc_name = sc_name
        if sc_name is not None:
            sc_w = state_dict[f"{prefix}{sc_name}.0.weight"]
            sc_b = f"{prefix}{sc_name}.0.bias" in state_dict
            sc_mod = nn.Sequential(
                nn.Conv2d(sc_w.shape[1], sc_w.shape[0], kernel_size=sc_w.shape[2], stride=stride, bias=sc_b),
                nn.BatchNorm2d(sc_w.shape[0]),
            )
            setattr(self, sc_name, sc_mod)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.sc_name is not None:
            identity = getattr(self, self.sc_name)(x)
        else:
            identity = x
        if identity.shape[2:] != out.shape[2:]:
            identity = F.adaptive_avg_pool2d(identity, out.shape[2:])
        return self.act(out + identity)


class CustomFASNet(nn.Module):
    """Kiến trúc CustomFASNet tự động tái tạo từ state_dict của Stage 1."""
    def __init__(self, state_dict: dict, theta: float = 0.4):
        super().__init__()
        stem_layers = []
        stem_w_keys = [k for k in state_dict.keys() if k.startswith("stem.") and k.endswith("weight")]
        stem_indices = sorted(set(int(k.split(".")[1]) for k in stem_w_keys))
        max_stem_idx = max(stem_indices) + 1 if stem_indices else 2

        for idx in range(max_stem_idx + 1):
            if f"stem.{idx}.conv.weight" in state_dict:
                w = state_dict[f"stem.{idx}.conv.weight"]
                stem_layers.append(Conv2d_CD(w.shape[1], w.shape[0], w.shape[2], stride=2, padding=1, theta=theta, direct_weight=False))
            elif f"stem.{idx}.weight" in state_dict:
                w = state_dict[f"stem.{idx}.weight"]
                if w.ndim == 4:
                    stem_layers.append(Conv2d_CD(w.shape[1], w.shape[0], w.shape[2], stride=2, padding=1, theta=theta, direct_weight=True))
                elif w.ndim == 1:
                    stem_layers.append(nn.BatchNorm2d(w.shape[0]))
            else:
                stem_layers.append(nn.SiLU(inplace=False))
        self.stem = nn.Sequential(*stem_layers)

        stage_names = sorted(set(
            k.split(".")[0] for k in state_dict.keys()
            if re.match(r"^(stage|block|layer)\d+$", k.split(".")[0])
        ))
        self.stage_names = stage_names
        for sname in stage_names:
            has_sc = any(k.startswith(f"{sname}.shortcut.") or k.startswith(f"{sname}.downsample.") for k in state_dict.keys())
            stride = 2 if has_sc else 1
            setattr(self, sname, DynamicResidualCDBlock(f"{sname}.", state_dict, theta=theta, stride=stride))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        cls_w_keys = [k for k in state_dict.keys() if k.startswith("classifier.") and k.endswith(".weight")]
        cls_indices = sorted(int(k.split(".")[1]) for k in cls_w_keys)
        max_cls_idx = max(cls_indices) if cls_indices else 0

        cls_layers = []
        for idx in range(max_cls_idx + 1):
            w_key = f"classifier.{idx}.weight"
            b_key = f"classifier.{idx}.bias"
            if w_key in state_dict:
                w = state_dict[w_key]
                if w.ndim == 2:
                    cls_layers.append(nn.Linear(w.shape[1], w.shape[0], bias=(b_key in state_dict)))
                elif w.ndim == 1:
                    cls_layers.append(nn.BatchNorm1d(w.shape[0]))
            else:
                if len(cls_layers) > 0 and isinstance(cls_layers[-1], (nn.BatchNorm1d, nn.Linear)):
                    next_is_bn = (f"classifier.{idx+1}.weight" in state_dict and state_dict[f"classifier.{idx+1}.weight"].ndim == 1)
                    cls_layers.append(nn.Identity() if next_is_bn else nn.SiLU(inplace=False))
                else:
                    cls_layers.append(nn.Identity())
        self.classifier = nn.Sequential(*cls_layers)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for sname in self.stage_names:
            x = getattr(self, sname)(x)
        return torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


# ==============================================================================
# 2. KIẾN TRÚC MODEL 2 (TransferFASNet - STAGE 2)
# ==============================================================================
def disable_inplace_activations(module: nn.Module):
    for m in module.modules():
        if hasattr(m, "inplace") and isinstance(m.inplace, bool):
            m.inplace = False


class TransferFASNet(nn.Module):
    def __init__(self, backbone_name: str = "mobilenet_v3_large", num_classes: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        if self.backbone_name == "mobilenet_v3_large":
            base = tv_models.mobilenet_v3_large(weights=None)
            self.encoder = base.features
            self.out_channels = 960
        elif self.backbone_name == "efficientnet_b0":
            base = tv_models.efficientnet_b0(weights=None)
            self.encoder = base.features
            self.out_channels = 1280
        elif self.backbone_name == "resnet34":
            base = tv_models.resnet34(weights=None)
            self.encoder = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )
            self.out_channels = 512
        else:
            raise ValueError(f"Không hỗ trợ backbone: {backbone_name}")

        disable_inplace_activations(self.encoder)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.feature_dim = self.out_channels * 2

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=False),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.encoder(x)
        feats = torch.cat([self.avg_pool(fmap).flatten(1), self.max_pool(fmap).flatten(1)], dim=1)
        return self.classifier(feats)


# ==============================================================================
# 3. NẠP TOÀN BỘ CHECKPOINT TRONG THƯ MỤC models/
# ==============================================================================
def get_compute_device(preferred: str = "XPU"):
    pref = preferred.upper()
    if pref in ("XPU", "AUTO", "NPU") and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0"), True, torch.bfloat16
    if pref in ("CUDA", "AUTO") and torch.cuda.is_available():
        return torch.device("cuda:0"), True, torch.float16
    return torch.device("cpu"), False, torch.float32


def load_models_from_dir(models_dir: Path, device: torch.device, use_channels_last: bool):
    if USER_CONFIG["load_only_best"]:
        ckpt_files = sorted(models_dir.glob("*_best.pt"))
    else:
        # Lấy các file *_best.pt và *_latest.pt chuẩn của Stage 1 & Stage 2
        ckpt_files = sorted(list(models_dir.glob("*_best.pt")) + list(models_dir.glob("*_latest.pt")))

    if not ckpt_files:
        raise FileNotFoundError(f"❌ Không tìm thấy file checkpoint nào trong {models_dir}")

    loaded_models = []
    for ckpt_path in ckpt_files:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            m_cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
            b_params = ckpt.get("best_params", {}) if isinstance(ckpt, dict) else {}

            raw_thr = float(ckpt.get("val_threshold", 0.3877 if "model1" in ckpt_path.name else 0.1987)) if isinstance(ckpt, dict) else 0.35
            val_thr = raw_thr
            if ckpt_path.name in USER_CONFIG["custom_model_thresholds"]:
                val_thr = float(USER_CONFIG["custom_model_thresholds"][ckpt_path.name])
            if USER_CONFIG["override_default_threshold"] is not None:
                val_thr = float(USER_CONFIG["override_default_threshold"])

            img_size = int(m_cfg.get("img_size", b_params.get("img_size", 192)))
            best_acer = ckpt.get("best_val_acer", None) if isinstance(ckpt, dict) else None
            if best_acer is not None and float(best_acer) < 1.0:
                best_acer = float(best_acer) * 100.0  # Đồng bộ đơn vị % giữa Stage 1 và Stage 2

            backbone_name = ckpt.get("backbone_name", m_cfg.get("backbone_name", b_params.get("backbone_name", None))) if isinstance(ckpt, dict) else None
            tag = "BEST" if "best" in ckpt_path.name else ("LATEST" if "latest" in ckpt_path.name else "CKPT")

            if backbone_name is not None or any(k.startswith("encoder.") for k in state_dict.keys()):
                bname = backbone_name or "mobilenet_v3_large"
                dropout = float(m_cfg.get("dropout_rate", b_params.get("dropout_rate", 0.4)))
                model = TransferFASNet(backbone_name=bname, num_classes=2, dropout_rate=dropout)
                display_name = f"Stage2-{tag}: {bname}"
            else:
                theta = float(m_cfg.get("theta", b_params.get("theta", 0.4)))
                model = CustomFASNet(state_dict=state_dict, theta=theta)
                display_name = f"Stage1-{tag}: CustomFASNet"

            model.load_state_dict(state_dict, strict=True)
            model.eval()

            # Chạy thử 1 forward nháp trên CPU để đảm bảo không có lỗi kích thước tensor
            with torch.no_grad():
                _ = model(torch.zeros(1, 3, img_size, img_size))

            if use_channels_last:
                model = model.to(device, memory_format=torch.channels_last)
            else:
                model = model.to(device)

            loaded_models.append({
                "file_name": ckpt_path.name,
                "display_name": display_name,
                "model": model,
                "backend_label": str(device),
                "threshold": val_thr,
                "default_threshold": raw_thr,
                "img_size": img_size,
                "best_val_acer": best_acer,
            })
            acer_str = f"{best_acer:.2f}%" if best_acer is not None else "N/A"
            print(f"✅ Đã nạp [{len(loaded_models)}]: {ckpt_path.name:28s} -> {display_name:28s} | τ* = {val_thr:.2f} (Gốc: {raw_thr:.4f}) | Val ACER = {acer_str}")
        except Exception as e:
            print(f"⚠️ Bỏ qua '{ckpt_path.name}': {e}")

    if not loaded_models:
        raise RuntimeError("❌ Không nạp được mô hình nào!")
    return loaded_models


# ==============================================================================
# 4. BỘ DÒ MẶT YUNET DNN & BÙ NÉT CHỈ KHI MẶT Ở XA (< 175px)
# ==============================================================================
class FaceDetectorLCC:
    YUNET_URL = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )

    def __init__(self, margin_ratio: float = 0.35, min_face_ratio: float = 0.12, max_missed_frames: int = 8):
        self.margin_ratio = margin_ratio
        self.min_face_ratio = min_face_ratio
        self.max_missed_frames = max_missed_frames
        self.smooth_box = None
        self.missed_count = 0
        self.yunet = self._init_yunet()
        self.face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.face_cascade_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    def _init_yunet(self):
        if not hasattr(cv2, "FaceDetectorYN"):
            return None
        model_path = Path(__file__).resolve().parent / "face_detection_yunet_2023mar.onnx"
        if not model_path.exists():
            try:
                import urllib.request
                urllib.request.urlretrieve(self.YUNET_URL, str(model_path))
            except Exception:
                return None
        try:
            return cv2.FaceDetectorYN.create(
                model=str(model_path), config="", input_size=(320, 320),
                score_threshold=0.60, nms_threshold=0.30, top_k=5000,
            )
        except Exception:
            return None

    def _detect_raw_box(self, frame_bgr: np.ndarray):
        h_img, w_img = frame_bgr.shape[:2]
        min_side = max(55, int(min(h_img, w_img) * self.min_face_ratio))

        if self.yunet is not None:
            det_w = 640
            scale = det_w / float(w_img)
            det_h = int(round(h_img * scale))
            small_bgr = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
            self.yunet.setInputSize((det_w, det_h))
            _, faces = self.yunet.detect(small_bgr)
            if faces is not None and len(faces) > 0:
                valid = [(int(f[0]/scale), int(f[1]/scale), int(f[2]/scale), int(f[3]/scale)) for f in faces if min(f[2]/scale, f[3]/scale) >= min_side]
                if valid:
                    return max(valid, key=lambda b: b[2] * b[3])

        scale_haar = 0.5 if w_img >= 960 else 1.0
        small_gray = cv2.equalizeHist(cv2.cvtColor(cv2.resize(frame_bgr, (0, 0), fx=scale_haar, fy=scale_haar), cv2.COLOR_BGR2GRAY))
        min_s = max(35, int(min_side * scale_haar))
        faces = self.face_cascade_default.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_s, min_s))
        if len(faces) == 0:
            faces = self.face_cascade_alt2.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_s, min_s))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            inv = 1.0 / scale_haar
            return int(x * inv), int(y * inv), int(w * inv), int(h * inv)
        return None

    @staticmethod
    def enhance_distant_face(face_rgb: np.ndarray, raw_crop_size: int, boost_strength: float) -> np.ndarray:
        """
        CHỈ kích hoạt bù nét khi khuôn mặt thực sự ở xa (raw_crop_size < 175px).
        Khi mặt hoặc điện thoại ở gần (>= 175px), trả về ảnh gốc 100%.
        """
        if boost_strength <= 0.01 or raw_crop_size >= 175:
            return face_rgb

        scale_deficit = min(1.0, (175.0 - raw_crop_size) / 95.0)
        amount = boost_strength * scale_deficit
        blurred = cv2.GaussianBlur(face_rgb, (0, 0), sigmaX=1.1)
        sharpened = cv2.addWeighted(face_rgb, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def detect_and_crop(self, frame_bgr: np.ndarray, target_size: int = 192, sharpness_boost: float = 0.4):
        h_img, w_img = frame_bgr.shape[:2]
        raw_box = self._detect_raw_box(frame_bgr)

        if raw_box is not None:
            self.missed_count = 0
            x, y, w, h = raw_box
            pad_w   = int(w * self.margin_ratio)
            pad_top = int(h * (self.margin_ratio * 1.20))
            pad_bot = int(h * (self.margin_ratio * 0.85))

            curr_box = np.array([
                max(0, x - pad_w), max(0, y - pad_top),
                min(w_img, x + w + pad_w), min(h_img, y + h + pad_bot)
            ], dtype=np.float32)
            self.smooth_box = curr_box if self.smooth_box is None else (0.60 * self.smooth_box + 0.40 * curr_box)
        else:
            self.missed_count += 1
            if self.smooth_box is None or self.missed_count > self.max_missed_frames:
                self.smooth_box = None
                return None, None, 0

        sx1, sy1, sx2, sy2 = self.smooth_box.astype(int)
        sx1, sy1, sx2, sy2 = max(0, sx1), max(0, sy1), min(w_img, sx2), min(h_img, sy2)
        face_crop_bgr = frame_bgr[sy1:sy2, sx1:sx2]
        if face_crop_bgr.size == 0:
            return None, None, 0

        raw_crop_size = min(sx2 - sx1, sy2 - sy1)
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        interp = cv2.INTER_CUBIC if raw_crop_size < target_size else cv2.INTER_AREA
        face_resized_rgb = cv2.resize(face_rgb, (target_size, target_size), interpolation=interp)
        face_resized_rgb = self.enhance_distant_face(face_resized_rgb, raw_crop_size, sharpness_boost)

        return face_resized_rgb, (sx1, sy1, sx2, sy2), raw_crop_size


@torch.no_grad()
def predict_liveness(
    model_info: dict, face_rgb: np.ndarray, device: torch.device,
    use_amp: bool, autocast_dtype: torch.dtype, use_channels_last: bool
) -> float:
    img_float = face_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    chw_np = np.ascontiguousarray(img_norm.transpose(2, 0, 1)[None, ...], dtype=np.float32)

    tensor = torch.from_numpy(chw_np)
    if use_channels_last:
        tensor = tensor.to(device, non_blocking=True, memory_format=torch.channels_last)
    else:
        tensor = tensor.to(device, non_blocking=True)

    device_type = "xpu" if device.type == "xpu" else ("cuda" if device.type == "cuda" else "cpu")
    with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_amp):
        logits = model_info["model"](tensor)
        prob_real = float(F.softmax(logits.float(), dim=-1)[0, 1].item())

    return prob_real


# ==============================================================================
# 5. VÒNG LẶP CHÍNH LIVE CAMERA
# ==============================================================================
def main():
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    if not models_dir.exists():
        models_dir = Path("models").resolve()

    device, use_amp, autocast_dtype = get_compute_device(USER_CONFIG["preferred_device"])
    use_channels_last = (device.type == "xpu")

    print("=" * 95)
    print(f"🚀 KHỞI ĐỘNG LIVE CAMERA FACE ANTI-SPOOFING | Thư mục: {models_dir}")
    print("=" * 95)

    models_list = load_models_from_dir(models_dir, device, use_channels_last)
    active_idx = next((i for i, m in enumerate(models_list) if "model2_transfer_best" in m["file_name"]), len(models_list) - 1)

    detector = FaceDetectorLCC(
        margin_ratio=USER_CONFIG["crop_margin_ratio"],
        min_face_ratio=USER_CONFIG["min_face_ratio"],
    )
    prob_history = deque(maxlen=USER_CONFIG["prob_smoothing_frames"])
    sharpness_boost = float(USER_CONFIG["distance_sharpness_boost"])
    show_crop_preview = bool(USER_CONFIG["show_crop_preview"])
    is_fullscreen = False

    cap = cv2.VideoCapture(USER_CONFIG["camera_id"])
    if not cap.isOpened():
        raise RuntimeError("❌ Không thể mở Webcam!")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, USER_CONFIG["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USER_CONFIG["camera_height"])
    cap.set(cv2.CAP_PROP_FPS, 30)

    window_name = "Face Anti-Spoofing Live (Multi-Model Switcher)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, USER_CONFIG["camera_width"], USER_CONFIG["camera_height"])

    def on_thr_change(val):
        models_list[active_idx]["threshold"] = max(0.01, min(0.99, val / 100.0))

    def on_margin_change(val):
        detector.margin_ratio = max(0.05, min(0.60, val / 100.0))

    def on_sharp_change(val):
        nonlocal sharpness_boost
        sharpness_boost = val / 100.0

    if USER_CONFIG["show_trackbars"]:
        cv2.createTrackbar("Threshold (%)", window_name, int(round(models_list[active_idx]["threshold"] * 100)), 95, on_thr_change)
        cv2.createTrackbar("Crop Margin (%)", window_name, int(round(detector.margin_ratio * 100)), 55, on_margin_change)
        cv2.createTrackbar("Dist Sharpness", window_name, int(round(sharpness_boost * 100)), 150, on_sharp_change)

    def sync_trackbar_to_model():
        if USER_CONFIG["show_trackbars"]:
            cv2.setTrackbarPos("Threshold (%)", window_name, int(round(models_list[active_idx]["threshold"] * 100)))

    prev_time = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h_disp, w_disp = frame.shape[:2]
        curr_info = models_list[active_idx]
        eff_threshold = float(curr_info["threshold"])

        t_infer_start = time.time()
        face_rgb, crop_box, raw_crop_px = detector.detect_and_crop(
            frame, target_size=curr_info["img_size"], sharpness_boost=sharpness_boost
        )

        if face_rgb is not None:
            raw_prob = predict_liveness(
                curr_info, face_rgb, device, use_amp, autocast_dtype, use_channels_last
            )
            prob_history.append(raw_prob)
            smooth_prob = float(np.mean(prob_history))
            infer_ms = (time.time() - t_infer_start) * 1000.0

            is_real = (smooth_prob >= eff_threshold)
            color = (0, 215, 0) if is_real else (0, 0, 235)
            label_str = "REAL (NGUOI THAT)" if is_real else "SPOOF (GIA MAO)"

            x1, y1, x2, y2 = crop_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            dist_status = f"Gan ({raw_crop_px}px - Goc)" if raw_crop_px >= 175 else f"Xa ({raw_crop_px}px - Boost)"
            header_text = f"{label_str} | P(Real)={smooth_prob:.2f} (Thr={eff_threshold:.2f}) | {dist_status}"
            (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
            bg_y1 = max(0, y1 - th - 14)
            cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 14, y1), color, -1)
            cv2.putText(
                frame, header_text, (x1 + 7, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA
            )

            if show_crop_preview:
                thumb_size = 160
                thumb_bgr = cv2.resize(cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR), (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
                px1, py1 = w_disp - thumb_size - 16, h_disp - thumb_size - 16
                frame[py1:h_disp - 16, px1:w_disp - 16] = thumb_bgr
                cv2.rectangle(frame, (px1, py1), (w_disp - 16, h_disp - 16), color, 2)
        else:
            prob_history.clear()
            infer_ms = 0.0
            cv2.putText(
                frame, "Khong phat hien khuon mat",
                (20, h_disp - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 215, 255), 2, cv2.LINE_AA
            )

        now = time.time()
        inst_fps = 1.0 / max(now - prev_time, 1e-5)
        prev_time = now
        fps_smooth = 0.85 * fps_smooth + 0.15 * inst_fps if fps_smooth > 0 else inst_fps

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_disp, 82), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        hud_line1 = (
            f"[{active_idx + 1}/{len(models_list)}] {curr_info['display_name']} ({curr_info['file_name']}) | "
            f"FPS: {fps_smooth:.1f} ({infer_ms:.1f}ms)"
        )
        hud_line2 = (
            f"Phim [1..{len(models_list)}/TAB]: Doi Model | [[/]]: Thr={eff_threshold:.2f} | "
            f"[,/.]: FarBoost={sharpness_boost:.2f} | [+/-]: Margin={int(detector.margin_ratio*100)}% | [F]: Fullscreen"
        )
        cv2.putText(frame, hud_line1, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(frame, hud_line2, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (225, 225, 225), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (9, ord("m"), ord("M")):
            active_idx = (active_idx + 1) % len(models_list)
            prob_history.clear()
            sync_trackbar_to_model()
            print(f"🔄 Đã chuyển sang [{active_idx+1}/{len(models_list)}]: {models_list[active_idx]['file_name']}")
        elif ord("1") <= key <= ord("9"):
            target_idx = key - ord("1")
            if target_idx < len(models_list):
                active_idx = target_idx
                prob_history.clear()
                sync_trackbar_to_model()
                print(f"🔄 Đã chuyển sang [{active_idx+1}/{len(models_list)}]: {models_list[active_idx]['file_name']}")
        elif key == ord("["):
            curr_info["threshold"] = max(0.02, curr_info["threshold"] - 0.02)
            sync_trackbar_to_model()
        elif key == ord("]"):
            curr_info["threshold"] = min(0.95, curr_info["threshold"] + 0.02)
            sync_trackbar_to_model()
        elif key in (ord("r"), ord("R")):
            curr_info["threshold"] = curr_info["default_threshold"]
            sync_trackbar_to_model()
        elif key in (ord("f"), ord("F")):
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
        elif key in (ord("p"), ord("P")):
            show_crop_preview = not show_crop_preview

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()