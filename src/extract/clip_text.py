import argparse
import json
import os
import time

import numpy as np
import torch
import open_clip

from .flickr30k import load_flickr30k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/flickr30k")
    ap.add_argument("--captions_file", default="captions.txt")

    ap.add_argument("--out_npy", default="features/track_b/clip_text.npy")
    ap.add_argument("--out_meta", default="features/track_b/clip_text_meta.json")

    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--limit", type=int, default=0, help="0 means all captions; >0 means only first N captions")
    args = ap.parse_args()

    docs, doc2img, _ = load_flickr30k(args.data_dir, captions_file=args.captions_file)
    if args.limit and args.limit > 0:
        docs = docs[:args.limit]
        doc2img = doc2img[:args.limit]

    os.makedirs(os.path.dirname(args.out_npy), exist_ok=True)

    device = torch.device(args.device)
    print(f"[CLIP-text] docs={len(docs)} model={args.model} pretrained={args.pretrained} device={device}")

    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)
    model.eval()

    # get embedding dim
    with torch.no_grad():
        dim = model.encode_text(tokenizer(["test"]).to(device)).shape[1]

    n = len(docs)
    tmp_path = args.out_npy + ".memmap"
    embs_mm = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(n, dim))

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            tokens = tokenizer(docs[start:end]).to(device)

            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize

            embs_mm[start:end] = feats.detach().cpu().numpy().astype(np.float32)
            embs_mm.flush()

            if start == 0 or (start // args.batch_size) % 200 == 0:
                print(f"[CLIP-text] encoded {end}/{n}")

    t1 = time.time()

    np.save(args.out_npy, np.asarray(embs_mm))
    try:
        del embs_mm
        os.remove(tmp_path)
    except Exception:
        pass

    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(
            {"doc2img": doc2img, "model": args.model, "pretrained": args.pretrained},
            f,
            ensure_ascii=False,
        )

    print(f"[CLIP-text] saved {args.out_npy} shape={(n, dim)} time={t1-t0:.2f}s")
    print(f"[CLIP-text] saved {args.out_meta}")


if __name__ == "__main__":
    main()
