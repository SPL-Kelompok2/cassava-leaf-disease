"""Uji dengan foto nyata: lapisan klasifikasi linear (linear probe) di atas VOLO-D2 384 beku.

BUKAN model 90,35 % milik laporan dan BUKAN pelatihan penuh seperti paper (tanpa fine-tuning dan
tanpa augmentasi). Tujuannya menguji jalur foto -> aplikasi dengan model yang benar-benar belajar
dari foto singkong, memakai bobot ImageNet resmi VOLO (github.com/sail-sg/volo) di CPU.

Pemakaian:
    python uji_foto_nyata.py --proyek DIR --data DIR --bobot FILE --fitur DIR --keluar DIR
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
for name in ("proyek", "data", "bobot", "fitur", "keluar"):
    p.add_argument("--" + name, required=True, type=Path)
args = p.parse_args()
sys.path.append(str(args.proyek.resolve()))

import timm  # noqa: E402
import torch  # noqa: E402

from cassava_es.data import (add_validation_split, classification_metrics, macro_f1,  # noqa: E402
                             normalize_label, stratified_split)
from cassava_es.dedup import deduplicate, find_duplicates  # noqa: E402
from cassava_es.rules import CLASSES  # noqa: E402
from cassava_es.vision import (IMAGENET_MEAN, IMAGENET_STD, VisionModel, preprocess,  # noqa: E402
                               save_checkpoint)

torch.manual_seed(2023)
torch.set_num_threads(2)
OUT = args.keluar
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------ fitur
X, paths = [], []
for f in sorted(args.fitur.glob("shard_*.npz")):
    z = np.load(f)
    X.append(z["X"])
    paths += [str(s) for s in z["paths"]]
X = np.concatenate(X).astype(np.float32)
assert len(set(paths)) == len(paths) == len(X)
df = pd.DataFrame({"image_id": paths})
df["folder"] = df.image_id.str.split("/").str[0]
df["label"] = [normalize_label(s.split("/")[1]) for s in paths]
df["y"] = [CLASSES.index(c) for c in df.label]
df["path"] = [str(args.data / s) for s in paths]
df["row"] = np.arange(len(df))
print("fitur:", X.shape)

# ------------------------------------------------------------ buang duplikat SEBELUM membagi
rep = find_duplicates(df, max_distance=4)
clean = deduplicate(rep, "near_group").copy()
print(f"citra: {len(rep)} -> setelah membuang duplikat/mirip: {len(clean)}")

# Data luar-lipatan: folder validation+test asli dataset, hanya grup yang SAMA SEKALI
# tidak menyentuh folder train.
touch_train = set(rep.loc[rep.folder == "train", "near_group"])
pool = clean[clean.near_group.isin(touch_train)].reset_index(drop=True)
external = clean[~clean.near_group.isin(touch_train)].reset_index(drop=True)
split = add_validation_split(stratified_split(pool, 0.2, 2023), 0.125, 2023)  # 70/10/20
parts = {k: split[split.split == k] for k in ("train", "val", "test")}
parts["luar"] = external
assert not (set(parts["train"].near_group) | set(parts["val"].near_group)) & (
    set(parts["test"].near_group) | set(parts["luar"].near_group))
for k, v in parts.items():
    print(f"{k:6s} {len(v):5d}", v.label.value_counts().reindex(list(CLASSES)).to_dict())

# ------------------------------------------------------------ latih lapisan linear
xt = torch.from_numpy(X)
mean = xt[parts["train"].row.values].mean(0)
std = xt[parts["train"].row.values].std(0).clamp_min(1e-6)


def data(part):
    idx = parts[part].row.values
    return (xt[idx] - mean) / std, torch.from_numpy(parts[part].y.values.copy()).long()


def fit(wd):
    xtr, ytr = data("train")
    head = torch.nn.Linear(xtr.shape[1], len(CLASSES))
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    opt = torch.optim.LBFGS(head.parameters(), lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(xtr), ytr) + wd * head.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return head


def metrics(head, part):
    x, y = data(part)
    with torch.inference_mode():
        prob = torch.softmax(head(x), 1)
    return classification_metrics(y.numpy(), prob.argmax(1).numpy()), prob.numpy()


grid = []
for wd in (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
    t = time.time()
    h = fit(wd)
    m, _ = metrics(h, "val")
    grid.append(dict(weight_decay=wd, val_accuracy=m["accuracy"], val_macro_f1=macro_f1(m)))
    print(f"wd={wd:g} val acc {m['accuracy']:.4f} macroF1 {grid[-1]['val_macro_f1']:.4f} ({time.time()-t:.0f}s)",
          flush=True)
best = max(grid, key=lambda g: g["val_macro_f1"])
head = fit(best["weight_decay"])
result = {"pemilihan": grid, "weight_decay_terpilih": best["weight_decay"], "jumlah": {}}
for k in ("train", "val", "test", "luar"):
    m, prob = metrics(head, k)
    m["macro_f1"] = macro_f1(m)
    result[k] = m
    result["jumlah"][k] = len(parts[k])
    pd.DataFrame(np.c_[parts[k].y.values, prob.argmax(1), prob],
                 columns=["y_true", "y_pred"] + [f"p_{c}" for c in CLASSES]).assign(
        image_id=parts[k].image_id.values).to_csv(OUT / f"prediksi_{k}.csv", index=False)
    print(f"== {k}: akurasi {m['accuracy']:.4f} macro F1 {m['macro_f1']:.4f} n={m['n']}")
majority = parts["train"].label.value_counts().idxmax()
for k in ("test", "luar"):
    result[k]["akurasi_tebak_mayoritas"] = float((parts[k].label == majority).mean())
    rng = np.random.RandomState(0)
    ok = (pd.read_csv(OUT / f"prediksi_{k}.csv").eval("y_true == y_pred")).values
    boot = [rng.choice(ok, len(ok)).mean() for _ in range(2000)]
    result[k]["akurasi_ci95_bootstrap"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
pd.concat([v.assign(split=k) for k, v in parts.items()])[["image_id", "label", "split"]].to_csv(
    OUT / "pembagian.csv", index=False)

# ------------------------------------------------------------ checkpoint untuk aplikasi
sd = torch.load(args.bobot, map_location="cpu", weights_only=True)
sd = sd.get("state_dict", sd)
model = timm.create_model("volo_d2_384", pretrained=False, num_classes=len(CLASSES))
backbone = {k: v for k, v in sd.items() if not k.startswith(("head.", "aux_head."))}
missing, unexpected = model.load_state_dict(backbone, strict=False)
assert sorted(missing) == ["aux_head.bias", "aux_head.weight", "head.bias", "head.weight"], missing
assert not unexpected
with torch.no_grad():
    # bobot untuk fitur terstandardisasi -> bobot untuk fitur mentah
    model.head.weight.copy_(head.weight / std)
    model.head.bias.copy_(head.bias - (head.weight / std) @ mean)
    model.aux_head.weight.zero_()  # timm menambah 0.5*aux_head(token) ; dinolkan
    model.aux_head.bias.zero_()
model.eval()
ck = OUT / "vision_linear_probe.pt"
save_checkpoint(ck, model, model_name="volo_d2_384", img_size=384, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                metrics={"accuracy": result["test"]["accuracy"]},
                meta={"jenis": "linear probe di atas VOLO-D2 384 beku (bobot ImageNet resmi); hanya untuk uji",
                      "weight_decay": best["weight_decay"], "n_train": len(parts["train"])})

# ------------------------------------------------------------ verifikasi: checkpoint lewat kode aplikasi
vm = VisionModel.load(ck, device="cpu")
ext = parts["luar"]
pred_saved = pd.read_csv(OUT / "prediksi_luar.csv").set_index("image_id")
rows, maxdiff, t0 = [], 0.0, time.time()
for i in range(0, len(ext), 4):
    chunk = ext.iloc[i:i + 4]
    for (_, r), pr in zip(chunk.iterrows(), vm.predict_proba_batch(list(chunk.path))):
        a = np.array([pr[c] for c in CLASSES])
        b = pred_saved.loc[r.image_id, [f"p_{c}" for c in CLASSES]].values.astype(float)
        maxdiff = max(maxdiff, float(np.abs(a - b).max()))
        rows.append((r.image_id, r.label, CLASSES[int(a.argmax())], *a))
sec = (time.time() - t0) / len(ext)
out_df = pd.DataFrame(rows, columns=["image_id", "benar", "prediksi"] + [f"p_{c}" for c in CLASSES])
out_df.to_csv(OUT / "prediksi_luar_lewat_aplikasi.csv", index=False)
acc = float((out_df.benar == out_df.prediksi).mean())
result["verifikasi_checkpoint"] = dict(n=len(ext), akurasi=acc, selisih_probabilitas_maks=maxdiff,
                                       detik_per_foto=sec)
print(f"verifikasi VisionModel: akurasi luar {acc:.4f}, selisih maks {maxdiff:.2e}, {sec:.2f} dtk/foto")
assert maxdiff < 1e-3 and abs(acc - result["luar"]["accuracy"]) < 1e-9
(OUT / "hasil_uji_foto_nyata.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
print("SELESAI")
