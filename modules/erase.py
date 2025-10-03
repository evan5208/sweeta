from typing import List
import cv2
import numpy as np
import torch
from modules import CONFIG
from modules.sttn import build_sttn_model, inpaint_video_with_builded_sttn
from utils.video_utils import detect_video_orientation

_model_cache = {}

def get_sttn_model(ckpt_p: str, device: str):
    if ckpt_p not in _model_cache:
        _model_cache[ckpt_p] = build_sttn_model(ckpt_p, device)
    return _model_cache[ckpt_p]

@torch.no_grad()
def inpaint_video_np(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    neighbor_stride: int,
    ckpt_p: str,
) -> List[np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_sttn_model(ckpt_p, device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype):
        return inpaint_video_with_builded_sttn(
            model, frames, masks, neighbor_stride, device, use_numpy=True
        )

def extract_masks_from_frames_fast(
    frame_shape: tuple,
    positions: List[List[int]],
    mask_expand: int = 20,
    n_frames: int = 1,
) -> List[np.ndarray]:
    h, w = frame_shape[:2]
    base_mask = np.zeros((h, w), dtype=np.uint8)
    for xmin, ymin, xmax, ymax in positions:
        cv2.rectangle(
            base_mask,
            (max(0, xmin - mask_expand), max(0, ymin - mask_expand)),
            (min(xmax + mask_expand, w - 1), min(ymax + mask_expand, h - 1)),
            255,
            thickness=-1,
        )
    return [base_mask.copy() for _ in range(n_frames)]

def remove_watermark_from_frames(
    frames: List, video_path: str
) -> List[np.ndarray]:
    orientation = detect_video_orientation(video_path)
    positions_key = "positions_portrait" if orientation == "portrait" else "positions_landscape"
    positions = CONFIG["watermark"].get(positions_key, [])
    if not positions:
        return frames
    if hasattr(frames[0], "shape"):
        frames_np = [f for f in frames]
    else:
        from PIL import Image
        frames_np = [np.array(f.convert("RGB")) for f in frames]
    masks_list = extract_masks_from_frames_fast(
        frames_np[0].shape, positions, CONFIG["watermark"]["mask_expand"], len(frames_np)
    )
    return inpaint_video_np(
        frames_np,
        masks_list,
        CONFIG["watermark"]["neighbor_stride"],
        CONFIG["watermark"]["ckpt_p"],
    )