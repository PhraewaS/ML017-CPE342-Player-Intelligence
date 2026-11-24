# CPE342 Task 4: train on your GPU (Windows, WSL or Linux)

Thanks for helping! This trains one image classifier and saves its predictions on the test images.
The file we need back is `out/clip_full_s0/test_logprob.npy` (about 0.5 MB).

## 1. Setup (once)
You need Python 3.10–3.12 and an NVIDIA GPU, ideally with 8 GB of VRAM or more.

**Windows (PowerShell), inside this folder:**
```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The last line must print `True` and your GPU name. If it prints `False`, reinstall torch with the CUDA build that
matches your driver (`cu121`, `cu124` or `cu126`; see https://pytorch.org/get-started/locally/).

**WSL / Linux:** the same commands, but activate with `source .venv/bin/activate`. Keep the images inside the Linux
filesystem (for example `~/t4data`), **not** under `/mnt/c/...`, which is very slow for 57k small files.

## 2. Images
Unzip `task4_images.zip` (the competition's Task 4 images, about 620 MB) somewhere, e.g. `data/`. The folder you pass
to `--data` must contain `train.csv`, `test.csv`, `train/` and `test/`. A `__MACOSX` folder in the zip is junk you can ignore.

## 3. Run (in this order)
```
python train_t4.py --data data --smoke      # about 2 minutes: checks everything works
python train_t4.py --data data --full       # THE IMPORTANT ONE: trains on all images, predicts the test set
python train_t4.py --data data --holdout    # optional: scores the model on the most test-like scenes
```
- `--full` trains a CLIP-pretrained ConvNeXt-Base for 12 epochs, on roughly 31.5k images per epoch. Expect about
  2–4 minutes per epoch on an RTX 3060–4080, so 30–50 minutes in total. The first run downloads about 350 MB of
  pretrained weights.
- **Out of GPU memory?** Add `--bs 16` (or `--bs 8`). The learning rate is scaled automatically.
- **Workers hang or crash on Windows?** Add `--workers 2` (or `--workers 0`, which is slower but always works).
- If you have spare time afterwards, any of these also helps, in this order:
  1. `--full --seed 1` (a second copy of the same model)
  2. `--preset convnext_small --full` (our previous recipe)
  3. `--cv` (5 folds, about 5× slower; it resumes if interrupted)

## 4. Send back
Zip the whole `out/` folder and send it (it's small). For each run it contains `test_logprob.npy`, `task4.csv`,
`log.txt` and `result.json`.

## Notes
- Competition rules: only Task 4 may use pretrained models (that is what this uses). Test images are only predicted,
  never trained on.
- `scene_groups.csv` (for `--cv`) and `testlike_holdout.csv` (for `--holdout`) are fixed splits of the training images.
