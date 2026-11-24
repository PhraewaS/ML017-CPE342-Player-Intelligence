"""Extract the three image views used by the Task 4 kernel model."""

import argparse
import io
import os
import zipfile
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="3", OMP_NUM_THREADS="3", MKL_NUM_THREADS="3")

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize

from shared.config import sample_ids
from .upright import upright


BATCH = 16
MODELS = {"dino": "dinov2-small.onnx", "upright": "dinov2-small.onnx", "clip": "clip-vision.onnx"}


def make_session(model, provider):
    providers = {
        "cpu": ["CPUExecutionProvider"],
        "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }[provider]
    if providers[0] not in ort.get_available_providers():
        raise RuntimeError(f"ONNX Runtime does not have {providers[0]}")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 2
    options.enable_mem_pattern = False
    return ort.InferenceSession(str(model), options, providers=providers)


def extract_embeddings(archive, split, names, view, session):
    is_clip = view == "clip"
    if is_clip:
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    else:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    features = np.empty((len(names), 512 if is_clip else 768), dtype=np.float32)
    for start in range(0, len(names), BATCH):
        images = []
        for name in names[start:start + BATCH]:
            data = archive.read(f"{split}/{name}")
            with Image.open(io.BytesIO(data)) as image:
                rgb = image.convert("RGB")
                if view == "upright":
                    rgb = upright(rgb)
                if is_clip:
                    rgb = ImageOps.autocontrast(rgb, cutoff=0.7)
                images.append(np.asarray(rgb.resize((224, 224), Image.Resampling.BICUBIC)))
        count = len(images)
        images.extend([images[-1]] * (BATCH - count))
        pixels = ((np.stack(images).astype(np.float32) / 255 - mean) / std)
        pixels = pixels.transpose(0, 3, 1, 2).copy()
        tokens = session.run(["image_embeds"] if is_clip else None, {"pixel_values": pixels})[0][:count]
        if is_clip:
            result = normalize(tokens)
        else:
            result = np.concatenate((normalize(tokens[:, 0]), normalize(tokens[:, 1:].mean(axis=1))), axis=1)
        features[start:start + count] = result
        if start % 4096 == 0:
            print(f"{view} {split}: {start + count:,}/{len(names):,}", flush=True)
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Path to task4_images.zip")
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("embeddings"))
    parser.add_argument("--provider", choices=["cpu", "directml", "cuda"], default="cpu")
    parser.add_argument("--views", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.images) as archive:
        train = pd.read_csv(archive.open("train.csv"))
        test = pd.read_csv(archive.open("test.csv"))
        if test.id.tolist() != sample_ids():
            raise ValueError("Task 4 test IDs do not match the sample submission order")
        for view in args.views:
            session = make_session(args.models / MODELS[view], args.provider)
            for split, frame in (("train", train), ("test", test)):
                features = extract_embeddings(archive, split, frame.file_name.tolist(), view, session)
                path = args.out / f"{view}-{split}.npy"
                np.save(path, features)
                print(f"Wrote {path}", flush=True)
            del session


if __name__ == "__main__":
    main()
