import sys
sys.path.insert(0, "./STTN")

from typing import List
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import cv2

from STTN.core.utils import Stack, ToTorchFormatTensor
from STTN.model import sttn

_to_tensors = transforms.Compose([Stack(), ToTorchFormatTensor()])


def get_ref_index(neighbor_ids, length):
    ref_length = 20
    ref_index = []
    for i in range(0, length, ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index


def build_sttn_model(ckpt_p: str, device="cuda"):
    model = sttn.InpaintGenerator().to(device)
    data = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(data["netG"])
    model.eval()
    return model


@torch.no_grad()
def inpaint_video_with_builded_sttn(
    model,
    frames: List,
    masks: List,
    neighbor_stride: int = 10,
    device="cuda",
    use_numpy: bool = False,
) -> List[np.ndarray]:
    w, h = 432, 240
    video_length = len(frames)

    if use_numpy:
        feats = [cv2.resize(f, (w, h)) for f in frames]
        feats = torch.from_numpy(np.stack(feats)).permute(0, 3, 1, 2).float() / 127.5 - 1
        _masks = [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in masks]
        _masks = torch.from_numpy(np.stack(_masks))[..., None].permute(0, 3, 1, 2)
    else:
        feats = [frame.resize((w, h)) for frame in frames]
        feats = _to_tensors(feats).unsqueeze(0) * 2 - 1
        _masks = [mask.resize((w, h), Image.NEAREST) for mask in masks]
        _masks = _to_tensors(_masks).unsqueeze(0)

    feats, _masks = feats.to(device), _masks.to(device)
    comp_frames = [None] * video_length

    feats = (feats * (1 - _masks).float()).view(video_length, 3, h, w)
    feats = model.encoder(feats)
    _, c, feat_h, feat_w = feats.size()
    feats = feats.view(1, video_length, c, feat_h, feat_w)

    for f in tqdm(range(0, video_length, neighbor_stride), desc="Inpaint Image", leave=False):
        neighbor_ids = list(range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)))
        ref_ids = get_ref_index(neighbor_ids, video_length)

        pred_feat = model.infer(
            feats[0, neighbor_ids + ref_ids, :, :, :],
            _masks[0, neighbor_ids + ref_ids, :, :, :],
        )
        pred_img = model.decoder(pred_feat[: len(neighbor_ids), :, :, :])
        pred_img = torch.tanh(pred_img)
        pred_img = (pred_img + 1) / 2
        pred_img = pred_img.permute(0, 2, 3, 1) * 255

        for i in range(len(neighbor_ids)):
            idx = neighbor_ids[i]
            b_mask = _masks.squeeze()[idx].unsqueeze(-1)
            b_mask = (b_mask != 0).int()
            if use_numpy:
                frame = torch.from_numpy(cv2.resize(frames[idx], (w, h))).to(device)
            else:
                frame = torch.from_numpy(np.array(frames[idx].resize((w, h)))).to(device)
            img = pred_img[i] * b_mask + frame * (1 - b_mask)
            img = img.cpu().numpy()
            if comp_frames[idx] is None:
                comp_frames[idx] = img
            else:
                comp_frames[idx] = comp_frames[idx] * 0.5 + img * 0.5

    result = []
    if use_numpy:
        ori_h, ori_w = frames[0].shape[:2]
    else:
        ori_w, ori_h = frames[0].size

    for idx in tqdm(range(len(frames)), desc="Restore Image", leave=False):
        if use_numpy:
            frame = frames[idx]
            b_mask = np.uint8(masks[idx][..., np.newaxis] != 0)
        else:
            frame = np.array(frames[idx])
            b_mask = np.uint8(np.array(masks[idx])[..., np.newaxis] != 0)

        comp_frame = np.uint8(comp_frames[idx])
        comp_frame = cv2.resize(comp_frame, (ori_w, ori_h))
        comp_frame = comp_frame * b_mask + frame * (1 - b_mask)
        result.append(comp_frame)

    return result
