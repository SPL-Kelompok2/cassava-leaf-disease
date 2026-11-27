#!/usr/bin/env python
"""Latih model visi (klasifikasi 5 kelas daun singkong) dari ``merged.csv``.

Mengikuti Bagian 4.5 + Tabel 5 paper:
    * pembagian 80 % latih / 20 % uji (distratifikasi per kelas, seed tetap);
    * transfer learning dari bobot ImageNet lewat ``timm`` (default VOLO-D2 384);
    * Adam, lr 1e-4, weight decay 1e-6, batch 16, 10 epoch;
    * CosineAnnealingWarmRestarts (T_0=10, T_mult=1, eta_min=1e-6);
    * Taylor Cross Entropy + label smoothing 0.05;
    * augmentasi: RandomResizedCrop, HorizontalFlip, Transpose, VerticalFlip,
      ShiftScaleRotate (p=0.5 masing-masing), Normalize ImageNet.

Contoh (meniru paper, butuh GPU besar; paper memakai Tesla V100 32 GB):

    python scripts/train_vision.py --csv data/merged.csv --img-dir data/train \\
        --out checkpoints/vision.pt

Contoh lebih ringan (GPU kecil/Colab), ganti arsitektur dan resolusi:

    python scripts/train_vision.py --csv data/merged.csv --img-dir data/train \\
        --model tf_efficientnetv2_s --img-size 384 --batch-size 16

Keluaran di folder yang sama dengan ``--out``:
    vision.pt               checkpoint epoch terakhir
    vision_metrics.json     konfigurasi, laporan dataset, riwayat epoch, metrik akhir
    vision_split.csv        pembagian latih/uji (pakai ulang untuk evaluasi fusi)
    vision_test_preds.csv   probabilitas per citra uji (untuk evaluasi fusi)

Catatan: checkpoint yang disimpan adalah epoch terakhir (tanpa pemilihan epoch berdasarkan
data uji); akurasi uji per epoch hanya dicetak untuk pemantauan.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cassava_es.data import (  # noqa: E402
    classification_metrics,
    load_merged_csv,
    stratified_split,
)
from cassava_es.rules import CLASSES  # noqa: E402
from cassava_es.vision import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess,
    save_checkpoint,
    to_pil,
)

# Distribusi kelas pada dataset paper (Gambar 7), hanya untuk pembanding visual.
PAPER_COUNTS = {"CBB": 1553, "CBSD": 3632, "CGM": 3159, "CMD": 15816, "Healthy": 2893}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="path merged.csv (kolom image_id, label, source)")
    p.add_argument("--img-dir", required=True, help="folder citra (datar, bukan per penyakit)")
    p.add_argument("--out", default="checkpoints/vision.pt", help="path checkpoint keluaran")
    p.add_argument("--model", default="volo_d2_384", help="nama model timm (paper: VOLO-D2 384)")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--smoothing", type=float, default=0.05)
    p.add_argument("--taylor-order", type=int, default=2)
    p.add_argument("--crop-scale-min", type=float, default=0.08, help="skala minimum RandomResizedCrop")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-pretrained", action="store_true", help="jangan unduh bobot ImageNet (uji cepat)")
    p.add_argument("--no-amp", action="store_true", help="matikan mixed precision (hanya berlaku di CUDA)")
    p.add_argument("--limit", type=int, default=None, help="pakai subset acak N citra (debug)")
    p.add_argument("--label-col", default="label")
    p.add_argument("--id-col", default="image_id")
    p.add_argument("--source-col", default="source")
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_datasets(df_train, df_test, args):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms as T

    mean, std = IMAGENET_MEAN, IMAGENET_STD
    train_tf = T.Compose([
        T.RandomResizedCrop(args.img_size, scale=(args.crop_scale_min, 1.0)),
        T.RandomHorizontalFlip(0.5),
        T.RandomApply([T.Lambda(lambda im: im.transpose(Image.TRANSPOSE))], p=0.5),  # Transpose
        T.RandomVerticalFlip(0.5),
        # ShiftScaleRotate (default Albumentations: shift 0.0625, scale 0.1, rotate 45)
        T.RandomApply([T.RandomAffine(degrees=45, translate=(0.0625, 0.0625), scale=(0.9, 1.1))], p=0.5),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    class TrainSet(Dataset):
        def __init__(self, df):
            self.paths, self.y = list(df["path"]), list(df["y"])

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return train_tf(to_pil(self.paths[i])), int(self.y[i])

    class EvalSet(Dataset):
        """Pra-pemrosesan = ``vision.preprocess`` (sama persis dengan inferensi)."""

        def __init__(self, df):
            self.paths, self.y = list(df["path"]), list(df["y"])

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return preprocess(self.paths[i], args.img_size, mean, std), int(self.y[i])

    return TrainSet(df_train), EvalSet(df_test)


def evaluate(model, loader, device, amp):
    import torch

    model.eval()
    ys, probs = [], []
    with torch.inference_mode():
        for x, y in loader:
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x.to(device, non_blocking=True))
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        import timm
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch/timm belum terpasang. Jalankan: pip install -r requirements-vision.txt", file=sys.stderr)
        return 2

    from cassava_es.losses import TaylorCrossEntropyLoss

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    seed_everything(args.seed)

    # ------------------------------------------------------------ data
    df, rep = load_merged_csv(
        args.csv, args.img_dir, id_col=args.id_col, label_col=args.label_col,
        source_col=args.source_col, limit=args.limit, seed=args.seed,
    )
    print(f"\nBaris CSV: {rep.total_rows} | citra terpakai: {rep.usable} | berkas hilang: {rep.missing_files}"
          f" | image_id ganda: {rep.duplicated_ids}")
    if rep.missing_files:
        print(f"  contoh berkas hilang: {rep.missing_examples}")
    print("Distribusi kelas (dataset | paper Gbr 7):")
    for c in CLASSES:
        print(f"  {c:8s} {rep.class_counts[c]:6d} | {PAPER_COUNTS[c]:6d}")
    if rep.source_counts:
        print(f"Sumber (tahun): {rep.source_counts}")
    print("  Catatan: CMD seharusnya kelas terbanyak; jika tidak, periksa pemetaan label.\n")
    if rep.usable == 0:
        print("Tidak ada citra yang bisa dimuat; periksa --img-dir dan kolom image_id.", file=sys.stderr)
        return 1

    df = stratified_split(df, args.test_size, args.seed)
    df[["image_id", "label", "source", "split"]].to_csv(f"{stem}_split.csv", index=False)
    df_train, df_test = df[df.split == "train"], df[df.split == "test"]
    print(f"Latih: {len(df_train)} | Uji: {len(df_test)}")

    train_set, test_set = make_datasets(df_train, df_test, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("PERINGATAN: tidak ada GPU CUDA. Pelatihan model besar di CPU sangat lambat; "
              "pakai GPU (mis. Colab/Kaggle) atau model/resolusi yang lebih kecil.")
    amp = device.type == "cuda" and not args.no_amp
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=len(train_set) > args.batch_size,
                              num_workers=args.num_workers, pin_memory=pin, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    # ------------------------------------------------------------ model
    try:
        model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=len(CLASSES))
    except Exception as exc:  # unduh bobot gagal / nama model salah
        print(f"Gagal membuat model '{args.model}' (pretrained={not args.no_pretrained}): {exc}", file=sys.stderr)
        return 1
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1, eta_min=1e-6)
    crit = TaylorCrossEntropyLoss(len(CLASSES), n=args.taylor_order, smoothing=args.smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    meta = dict(
        args={k: v for k, v in vars(args).items()}, dataset=rep.__dict__,
        n_train=len(df_train), n_test=len(df_test),
        torch=str(torch.__version__), timm=str(timm.__version__), device=str(device), amp=bool(amp),
    )

    # ------------------------------------------------------------ latih
    history, t_start = [], time.time()
    print(f"\nMelatih {args.model} @ {args.img_size}px, {args.epochs} epoch, batch {args.batch_size}, "
          f"perangkat {device}, AMP={amp}\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
            loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach()) * len(y)
            seen += len(y)
            if step % 50 == 0 or step == len(train_loader):
                print(f"  epoch {epoch}/{args.epochs} langkah {step}/{len(train_loader)} loss {running / seen:.4f}", end="\r")
        sched.step()
        y_true, probs = evaluate(model, test_loader, device, amp)
        m = classification_metrics(y_true, probs.argmax(1))
        history.append(dict(epoch=epoch, train_loss=running / max(seen, 1), test_accuracy=m["accuracy"],
                            lr=opt.param_groups[0]["lr"], seconds=time.time() - t0))
        print(f"\nepoch {epoch}/{args.epochs}: loss {running / max(seen, 1):.4f} | akurasi uji {m['accuracy']:.4f} "
              f"| {time.time() - t0:.0f} dtk")
        # simpan tiap epoch (menimpa): hasil akhir = epoch terakhir, bukan "terbaik di data uji"
        save_checkpoint(out, model, model_name=args.model, img_size=args.img_size,
                        mean=IMAGENET_MEAN, std=IMAGENET_STD, metrics=dict(accuracy=m["accuracy"], epoch=epoch), meta=meta)

    # ------------------------------------------------------------ akhir
    final = classification_metrics(y_true, probs.argmax(1))
    meta.update(history=history, total_seconds=time.time() - t_start)
    with open(f"{stem}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, final_test_metrics=final), f, indent=2, ensure_ascii=False)
    with open(f"{stem}_test_preds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "y_true", "y_pred"] + [f"p_{c}" for c in CLASSES])
        for iid, yt, pr in zip(df_test["image_id"], y_true, probs):
            w.writerow([iid, int(yt), int(pr.argmax())] + [f"{v:.6f}" for v in pr])

    print(f"\n=== Hasil akhir (epoch {args.epochs}, data uji {final['n']} citra) ===")
    print(f"Akurasi: {final['accuracy']:.4f}   (paper, VOLO-D2@384: 0.8964)")
    print(f"{'kelas':8s} {'presisi':>8s} {'recall':>8s} {'F1':>8s} {'n':>6s}")
    for c in CLASSES:
        r = final["per_class"][c]
        print(f"{c:8s} {r['precision']:8.3f} {r['recall']:8.3f} {r['f1']:8.3f} {r['support']:6d}")
    print(f"\nCheckpoint: {out}\nMetrik    : {stem}_metrics.json")
    print("Muat di app: sidebar -> 'Model visi' -> isi path checkpoint di atas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
