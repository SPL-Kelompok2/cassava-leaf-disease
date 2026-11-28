#!/usr/bin/env python
"""Periksa citra ganda pada dataset sebelum (atau sesudah) membagi latih/uji.

Contoh:
    # periksa dataset dan buat CSV bersih (membuang salinan identik) untuk pelatihan
    python scripts/check_duplicates.py --csv data/merged.csv --img-dir data/train \\
        --report duplikat.csv --clean-out data/merged_bersih.csv

    # periksa apakah pembagian yang sudah dipakai bocor (kolom split)
    python scripts/check_duplicates.py --csv data/merged.csv --img-dir data/train \\
        --split checkpoints/vision_split.csv

Kode keluar: 0 = selesai (ada atau tidak ada duplikat), 1 = tidak ada citra yang bisa dimuat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from cassava_es.data import load_merged_csv  # noqa: E402
from cassava_es.dedup import deduplicate, find_duplicates, summarize  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="path merged.csv")
    p.add_argument("--img-dir", required=True, help="folder citra")
    p.add_argument("--split", help="vision_split.csv (kolom image_id, split) untuk memeriksa kebocoran")
    p.add_argument("--max-distance", type=int, default=4, help="ambang jarak Hamming dHash untuk 'mirip'")
    p.add_argument("--report", help="tulis daftar citra yang masuk grup ganda ke CSV ini")
    p.add_argument("--clean-out", help="tulis CSV tanpa salinan ganda ke path ini")
    p.add_argument("--clean-near", action="store_true",
                   help="--clean-out juga membuang citra 'mirip' (periksa laporannya dulu)")
    p.add_argument("--label-col", default="label")
    p.add_argument("--id-col", default="image_id")
    p.add_argument("--source-col", default="source")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    df, rep = load_merged_csv(args.csv, args.img_dir, id_col=args.id_col, label_col=args.label_col,
                              source_col=args.source_col)
    print(f"Baris CSV: {rep.total_rows} | citra terpakai: {rep.usable} | berkas hilang: {rep.missing_files}"
          f" | image_id ganda: {rep.duplicated_ids}")
    if rep.usable == 0:
        print("Tidak ada citra yang bisa dimuat; periksa --img-dir dan kolom image_id.", file=sys.stderr)
        return 1
    if args.split:
        split = pd.read_csv(args.split, dtype={"image_id": str}).drop_duplicates("image_id")
        df = df.merge(split[["image_id", "split"]], on="image_id", how="left")
        df["split"] = df["split"].fillna("(tidak ada di split)")

    report = find_duplicates(df, args.max_distance)
    for col, title in (("exact_group", "IDENTIK (SHA-256)"),
                       ("near_group", f"IDENTIK atau MIRIP (dHash <= {args.max_distance}; perlu dicek mata)")):
        s = summarize(report, col)
        print(f"\n{title}\n  grup ganda: {s['groups']} | citra di dalamnya: {s['images']}"
              f" | salinan berlebih: {s['extra_copies']} | grup berlabel bentrok: {s['label_conflicts']}")
        if "cross_split_groups" in s:
            print(f"  grup yang menyeberang latih/validasi/uji (kebocoran): {s['cross_split_groups']}")

    cols = [c for c in ("image_id", "label", "source", "split", "sha256", "dhash", "exact_group", "near_group")
            if c in report.columns]
    if args.report:
        multi = report[report.groupby("near_group")["near_group"].transform("size") > 1]
        multi.sort_values(["near_group", "image_id"])[cols].to_csv(args.report, index=False)
        print(f"\nLaporan grup ganda: {args.report} ({len(multi)} baris)")
    if args.clean_out:
        group = "near_group" if args.clean_near else "exact_group"
        clean = deduplicate(report, group)
        clean[["image_id", "label", "source"]].to_csv(args.clean_out, index=False)
        print(f"CSV bersih ({group}): {args.clean_out} | {len(report)} -> {len(clean)} citra")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
