import os
import shutil
from pathlib import Path
def cleanup_checkpoints(ckpt_dir, prefix, keep=3):
    checkpoints = [
        d for d in os.listdir(ckpt_dir)
        if d.startswith(prefix)
        and os.path.isdir(os.path.join(ckpt_dir, d))
    ]

    checkpoints.sort(
        key=lambda x: int(x.rsplit("-", 1)[-1]),
        reverse=True
    )

    for ckpt in checkpoints[keep:]:
        shutil.rmtree(os.path.join(ckpt_dir, ckpt))


def get_latest_checkpoint(ckpt_dir, prefix="checkpoint-"):
    checkpoints = [
        p for p in Path(ckpt_dir).glob(f"{prefix}*")
        if p.is_dir() and p.name[len(prefix):].isdigit()
    ]

    if not checkpoints:
        return None, 0

    latest = max(
        checkpoints,
        key=lambda p: int(p.name[len(prefix):])
    )

    step = int(latest.name[len(prefix):])

    return latest, step