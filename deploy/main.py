"""Live face anti-spoofing demo.

Run from the repository root with:

	python deploy/main.py

Keys while the camera is open:
	1 - load model1_custom_cnn_best.pt
	2 - load model1_custom_cnn_latest.pt
	q - quit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import torch
from torch import Tensor, nn
from torch.nn import functional as F


IMAGE_SIZE = 192
REAL_THRESHOLD = 0.1
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


class Conv2dCD(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		stride: int = 1,
		theta: float = 0.7,
	) -> None:
		super().__init__()
		self.conv = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size=3,
			stride=stride,
			padding=1,
			bias=False,
		)
		self.theta = theta

	def forward(self, inputs: Tensor) -> Tensor:
		weights = self.conv.weight
		kernel_diff = weights.sum(dim=(2, 3))
		fused_weights = weights.clone()
		fused_weights[:, :, 1, 1] = weights[:, :, 1, 1] - self.theta * kernel_diff
		return F.conv2d(
			inputs,
			fused_weights,
			self.conv.bias,
			self.conv.stride,
			self.conv.padding,
			self.conv.dilation,
			self.conv.groups,
		)


class SEBlock(nn.Module):
	def __init__(self, channels: int, reduction: int = 8) -> None:
		super().__init__()
		middle_channels = max(channels // reduction, 8)
		self.fc = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(channels, middle_channels, bias=False),
			nn.SiLU(inplace=True),
			nn.Linear(middle_channels, channels, bias=False),
			nn.Sigmoid(),
		)

	def forward(self, inputs: Tensor) -> Tensor:
		weights = self.fc(inputs).unsqueeze(-1).unsqueeze(-1)
		return inputs * weights


class ResidualCDBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, stride: int, theta: float) -> None:
		super().__init__()
		self.conv1 = Conv2dCD(in_channels, out_channels, stride, theta)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.act = nn.SiLU(inplace=True)
		self.conv2 = Conv2dCD(out_channels, out_channels, 1, theta)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.se = SEBlock(out_channels)
		self.shortcut = (
			nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
				nn.BatchNorm2d(out_channels),
			)
			if stride != 1 or in_channels != out_channels
			else nn.Identity()
		)

	def forward(self, inputs: Tensor) -> Tensor:
		residual = self.shortcut(inputs)
		outputs = self.act(self.bn1(self.conv1(inputs)))
		outputs = self.bn2(self.conv2(outputs))
		outputs = self.se(outputs)
		return self.act(outputs + residual)


class CustomFASNet(nn.Module):
	def __init__(self, base_channels: int, dropout_rate: float, theta: float, num_classes: int = 2) -> None:
		super().__init__()
		channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
		self.stem = nn.Sequential(
			Conv2dCD(3, channels[0], stride=2, theta=theta),
			nn.BatchNorm2d(channels[0]),
			nn.SiLU(inplace=True),
		)
		self.stage1 = ResidualCDBlock(channels[0], channels[0], 1, theta)
		self.stage2 = ResidualCDBlock(channels[0], channels[1], 2, theta)
		self.stage3 = ResidualCDBlock(channels[1], channels[2], 2, theta)
		self.stage4 = ResidualCDBlock(channels[2], channels[3], 2, theta)
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)
		feature_dim = channels[3] * 2
		self.classifier = nn.Sequential(
			nn.Linear(feature_dim, channels[3], bias=False),
			nn.BatchNorm1d(channels[3]),
			nn.SiLU(inplace=True),
			nn.Dropout(dropout_rate),
			nn.Linear(channels[3], num_classes),
		)

	def forward(self, inputs: Tensor) -> Tensor:
		outputs = self.stem(inputs)
		outputs = self.stage1(outputs)
		outputs = self.stage2(outputs)
		outputs = self.stage3(outputs)
		outputs = self.stage4(outputs)
		features = torch.cat(
			(self.avg_pool(outputs).flatten(1), self.max_pool(outputs).flatten(1)),
			dim=1,
		)
		return self.classifier(features)


def select_device() -> torch.device:
	if hasattr(torch, "xpu") and torch.xpu.is_available():
		return torch.device("xpu")
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
	checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	config = checkpoint.get("model_config")
	if not isinstance(config, dict):
		raise ValueError(f"{checkpoint_path} does not contain model_config")
	model = CustomFASNet(**config).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	return model


def preprocess(face_bgr: Any, device: torch.device) -> Tensor:
	face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
	tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
	return ((tensor - IMAGENET_MEAN) / IMAGENET_STD).to(device)


def largest_face(faces: Any) -> tuple[int, int, int, int] | None:
	if len(faces) == 0:
		return None
	return max(faces, key=lambda box: int(box[2]) * int(box[3]))


def run(camera_index: int, checkpoint_paths: dict[str, Path]) -> None:
	device = select_device()
	detector = cv2.CascadeClassifier(
		str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
	)
	if detector.empty():
		raise RuntimeError("Could not load the OpenCV Haar face detector")

	models: dict[str, nn.Module] = {}
	active_name = next(iter(checkpoint_paths))
	capture = cv2.VideoCapture(camera_index)
	if not capture.isOpened():
		raise RuntimeError(f"Could not open camera index {camera_index}")

	try:
		while True:
			ok, frame = capture.read()
			if not ok:
				break

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
			face_box = largest_face(faces)
			if face_box is not None:
				x, y, width, height = face_box
				face = cv2.resize(frame[y : y + height, x : x + width], (IMAGE_SIZE, IMAGE_SIZE))
				if active_name not in models:
					models[active_name] = load_model(checkpoint_paths[active_name], device)
				with torch.inference_mode():
					probabilities = models[active_name](preprocess(face, device)).softmax(dim=1)[0]
				real_probability = float(probabilities[1].item())
				label = "REAL" if real_probability >= REAL_THRESHOLD else "SPOOF"
				color = (0, 180, 0) if label == "REAL" else (0, 0, 220)
				cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
				cv2.putText(
					frame,
					f"{label}  real={real_probability:.2f}",
					(x, max(28, y - 10)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					color,
					2,
					cv2.LINE_AA,
				)

			cv2.putText(frame, f"Model [{active_name}]  device={device}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
			cv2.putText(frame, "1/2: change model   q: quit", (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
			cv2.imshow("Face Anti-Spoofing Live", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
			if key == ord("1") and "best" in checkpoint_paths:
				active_name = "best"
			elif key == ord("2") and "latest" in checkpoint_paths:
				active_name = "latest"
	finally:
		capture.release()
		cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run live face anti-spoofing inference")
	parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
	parser.add_argument("--best", type=Path, default=Path("models/model1_custom_cnn_best.pt"))
	parser.add_argument("--latest", type=Path, default=Path("models/model1_custom_cnn_latest.pt"))
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(args.camera, {"best": args.best, "latest": args.latest})