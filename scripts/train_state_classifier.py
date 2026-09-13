"""Train the camera-only task-state classifier on the table demos and export it to OpenVINO.

    python scripts/train_state_classifier.py --cache data/table_v1_skill_cache --out models/state_classifier

Training reads the top camera straight from the fastdata memmap cache. The last 10 oracle runs (50 skill
episodes) are held out for validation; the report gives per-skill accuracy on them.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.state_classifier import N_SKILLS, build_model, labels_from_structure, preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="fastdata cache: labels from the demo structure")
    ap.add_argument("--labelled", default=None, help="dir of shard_*.npz from record_state_data.py (simulator-truth labels)")
    ap.add_argument("--out", default="models/state_classifier")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-runs", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.labelled:
        shards = sorted(Path(args.labelled).glob("shard_*.npz"))
        data = [np.load(f) for f in shards]
        imgs = np.concatenate([d["images"] for d in data])
        y = np.concatenate([d["labels"] for d in data]).astype(np.float32)
        # Hold out the last shard (whole runs) for validation.
        n_last = len(data[-1]["labels"])
        tr_idx, va_idx = np.arange(len(y) - n_last), np.arange(len(y) - n_last, len(y))
    else:
        cache = Path(args.cache)
        imgs = np.load(cache / "observation__images__top.npy", mmap_mode="r")
        z = np.load(cache / "vectors.npz")
        ep, fr = z["episode_index"], z["frame_index"]
        y = labels_from_structure(ep, fr)
        val = ep >= (ep.max() + 1 - args.val_runs * N_SKILLS)
        tr_idx, va_idx = np.where(~val)[0], np.where(val)[0]
    print(f"train {len(tr_idx)} frames, val {len(va_idx)} frames, positive rate {y.mean(0).round(3).tolist()}", flush=True)

    net = build_model(pretrained=True).to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    # Rare 'done' labels (e.g. the cup) get proportionally more weight.
    pos = torch.from_numpy(y[tr_idx].mean(0)).clamp(1e-3, 1 - 1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).clamp(max=20).to(args.device))
    rng = np.random.default_rng(0)

    def batch(idx):
        idx = np.sort(idx)  # sorted reads are much faster on a memmap
        x = preprocess(torch.from_numpy(np.ascontiguousarray(imgs[idx]))).to(args.device)
        return x, torch.from_numpy(y[idx]).to(args.device)

    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), args.batch):
            x, t = batch(perm[i:i + args.batch])
            loss = loss_fn(net(x), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (i // args.batch) % 200 == 0:
                print(f"epoch {epoch} step {i // args.batch} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)

    net.eval()
    correct, tp, pos_n, n = np.zeros(N_SKILLS), np.zeros(N_SKILLS), np.zeros(N_SKILLS), 0
    with torch.no_grad():
        for i in range(0, len(va_idx), 256):
            x, t = batch(va_idx[i:i + 256])
            pred = (torch.sigmoid(net(x)) > 0.5).float()
            correct += (pred == t).float().sum(0).cpu().numpy()
            tp += (pred * t).sum(0).cpu().numpy()
            pos_n += t.sum(0).cpu().numpy()
            n += len(t)
    acc = (correct / n).round(4).tolist()
    recall = (tp / np.maximum(pos_n, 1)).round(4).tolist()
    print("validation accuracy per skill (drawer, spoon, plate, fork, cup):", acc, flush=True)
    print("validation recall of 'done' per skill:", recall, flush=True)

    import openvino as ov

    net_cpu = net.cpu().eval()
    example = torch.zeros(1, 3, *imgs.shape[2:])
    model = ov.convert_model(net_cpu, example_input=example)
    ov.save_model(model, out / "state_classifier.xml")
    torch.save(net_cpu.state_dict(), out / "state_classifier.pt")
    (out / "report.json").write_text(json.dumps({"val_accuracy": acc, "val_recall_done": recall, "train_frames": int(len(tr_idx)),
                                                 "val_frames": int(len(va_idx)), "epochs": args.epochs,
                                                 "image_hw": list(imgs.shape[2:])}, indent=1))
    print("saved", out / "state_classifier.xml")


if __name__ == "__main__":
    main()
