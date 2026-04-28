"""
aurexis_hf.py — HuggingFace Datasets streaming for bulk ingestion.

The right way to train at scale on a laptop. Each call opens a remote
shard pipeline; samples decode and yield as you ask for them.
No full dataset is downloaded; HF caches what you've actually seen.

Datasets that work without authentication:
  - cifar10, cifar100              (60K images each, tiny but diverse)
  - food101                        (101K food images, 101 classes)
  - oxford-iiit-pet                (7K cats + dogs, fine-grained)
  - beans                          (1.3K plant disease)
  - cats_vs_dogs                   (23K)
  - keremberke/painting-style-classification (5K paintings, style labels)
  - huggan/wikiart                 (80K paintings)
  - Multimodal-Fatima/Caltech101_train (8K varied)
  - sasha/dog-food                 (2K)
  - frgfm/imagenette               (13K, ImageNet subset)
  - Matthijs/snacks                (6K)

Datasets that require a free HuggingFace token (set HF_TOKEN env var):
  - imagenet-1k                    (1.2M images, the canonical benchmark)
  - laion/laion400m                (400M image-text pairs)
  - kakaobrain/coyo-700m           (700M image-text pairs)

Get a token from huggingface.co/settings/tokens and set:
    setx HF_TOKEN your_token_here    (Windows persistent)
"""

import os
import time
from pathlib import Path
from PIL import Image


# ============================================================
# HF DATASETS REGISTRY
# ============================================================

# Each entry: (dataset_path, config_or_None, split, image_field, needs_auth)
HF_DATASETS = [
    # No auth needed
    ("cifar10",               None,    "train", "img",   False),
    ("cifar100",              None,    "train", "img",   False),
    ("food101",               None,    "train", "image", False),
    ("frgfm/imagenette",      "320px", "train", "image", False),
    ("beans",                 None,    "train", "image", False),
    ("cats_vs_dogs",          None,    "train", "image", False),
    ("Matthijs/snacks",       None,    "train", "image", False),
    ("sasha/dog-food",        None,    "train", "image", False),
    ("keremberke/painting-style-classification", "full",
                                       "train", "image", False),
    ("huggan/wikiart",        None,    "train", "image", False),
    ("Multimodal-Fatima/Caltech101_train", None,
                                       "train", "image", False),
    # Auth needed
    ("imagenet-1k",           None,    "train", "image", True),
    ("zh-plus/tiny-imagenet", None,    "train", "image", False),
]


def has_hf_token():
    """Check if an HF token is available — env var or _hf_token.txt file."""
    if os.environ.get("HF_TOKEN", "").strip():
        return True
    if os.environ.get("HUGGINGFACE_TOKEN", "").strip():
        return True
    # Check for a file-based token next to the archive folder
    for path in [Path("_hf_token.txt"), Path("aurexis_archive/_hf_token.txt")]:
        if path.exists():
            try:
                if path.read_text(encoding="utf-8").strip():
                    return True
            except Exception:
                pass
    return False


def _auth_kwargs():
    """Get HF auth args from env var or token file."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok:
        for path in [Path("_hf_token.txt"),
                      Path("aurexis_archive/_hf_token.txt")]:
            if path.exists():
                try:
                    tok = path.read_text(encoding="utf-8").strip()
                    if tok: break
                except Exception:
                    pass
    return {"token": tok} if tok else {}


# ============================================================
# STREAMING GENERATOR
# ============================================================

def stream_hf(dataset_path, config=None, split="train",
              image_field="image", n=200, label_field="label"):
    """Stream n images from a HuggingFace dataset.

    Yields (PIL.Image, alias, source) — drop-in compatible with sources.py.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("    [hf] 'datasets' library not installed. "
              "Run: pip install datasets")
        return

    src_name = f"hf_{dataset_path.split('/')[-1]}"
    print(f"    [hf] opening stream: {dataset_path}"
          + (f" ({config})" if config else ""))

    try:
        kwargs = dict(streaming=True, split=split, **_auth_kwargs())
        if config:
            ds = load_dataset(dataset_path, config, **kwargs)
        else:
            ds = load_dataset(dataset_path, **kwargs)
    except Exception as e:
        print(f"    [hf] failed to open {dataset_path}: "
              f"{type(e).__name__}: {str(e)[:100]}")
        return

    # Optionally shuffle the stream (uses a small reservoir buffer)
    try:
        ds = ds.shuffle(seed=int(time.time()) % (1 << 30),
                        buffer_size=1000)
    except Exception:
        pass

    fetched = 0
    try:
        for i, ex in enumerate(ds):
            if fetched >= n:
                break
            # Try the named field, fall back to common alternatives
            img = None
            for fname in (image_field, "image", "img", "picture"):
                if fname in ex and ex[fname] is not None:
                    img = ex[fname]
                    break
            if img is None:
                continue

            # HF returns PIL images directly in most cases
            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(img)
                except Exception:
                    continue

            try:
                img = img.convert("RGB")
            except Exception:
                continue

            # Build a useful alias including the label if present
            label = ex.get(label_field, None)
            label_part = f"_{label}" if label is not None else ""
            alias = f"{src_name}_{i:06d}{label_part}"
            yield img, alias, src_name
            fetched += 1
    except Exception as e:
        print(f"    [hf] stream error after {fetched}: "
              f"{type(e).__name__}: {str(e)[:100]}")

    print(f"    [hf] finished {dataset_path}: {fetched} images")


# ============================================================
# REGISTRY ENTRIES — for the CLI menu
# ============================================================

HF_SOURCES = []
for path, config, split, field, needs_auth in HF_DATASETS:
    short = path.split("/")[-1]
    label = f"HF: {short}" + (f" [{config}]" if config else "")
    key_req = "HF_TOKEN" if needs_auth else "no key"
    # bind args via default-arg trick to capture per-iteration values
    def _make_fn(p=path, c=config, s=split, f=field):
        return lambda n: stream_hf(p, c, s, f, n)
    HF_SOURCES.append((label, key_req, _make_fn()))
