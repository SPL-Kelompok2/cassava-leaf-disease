"""Ekstrak fitur token-kelas VOLO-D2@384 (bobot ImageNet resmi) untuk foto iCassava 2019 berlabel.

Pemakaian: python ekstrak_fitur.py DIR_PROYEK BOBOT_VOLO DIR_DATA DIR_FITUR
"""
import sys, time, pathlib, numpy as np
sys.path.append(sys.argv[1])  # salinan proyek (untuk preprocess yang sama dengan aplikasi)
import torch, timm
from cassava_es.vision import preprocess, IMAGENET_MEAN, IMAGENET_STD
weights, data, out = sys.argv[2], pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]); out.mkdir(exist_ok=True)
torch.set_num_threads(2)
sd = torch.load(weights, map_location="cpu", weights_only=True); sd = sd.get("state_dict", sd)
m = timm.create_model("volo_d2_384", pretrained=False, num_classes=1000); m.load_state_dict(sd, strict=True); m.eval()
files = []
for split in ("test", "validation", "train"):
    for c in ("cbb", "cbsd", "cgm", "cmd", "healthy"):
        files += sorted((data / split / c).glob("*.jpg"))
print("jumlah citra", len(files), flush=True)
B, S, t0 = 4, 200, time.time()
for s in range(0, len(files), S):
    f = out / f"shard_{s:05d}.npz"
    if f.exists(): continue
    chunk = files[s:s + S]; feats = []
    for i in range(0, len(chunk), B):
        x = torch.stack([preprocess(p, 384, IMAGENET_MEAN, IMAGENET_STD) for p in chunk[i:i + B]])
        with torch.inference_mode():
            feats.append(m.forward_features(x)[:, 0].numpy())
    np.savez(f, X=np.concatenate(feats), paths=np.array([str(p.relative_to(data)) for p in chunk]))
    done = s + len(chunk); print(f"{done}/{len(files)} {time.time()-t0:.0f}s", flush=True)
print("SELESAI", flush=True)
