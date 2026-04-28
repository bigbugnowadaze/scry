"""
aurexis_aws.py — AWS Open Data S3 streamers for high-throughput training.

When you run aurexis on EC2 in us-east-1, reading from these S3 buckets
costs nothing in egress and runs at gigabit+ speeds. ~10-50x faster than
hitting public APIs from a laptop.

All buckets here allow ANONYMOUS access — no AWS credentials needed for
the data itself, just the EC2 instance to read from.

Datasets included:
  - aft-vbi-pds          Amazon Bin Image Dataset (530K product images)
  - open-images-dataset  Google Open Images V7 (1.7M labeled images)
  - multimedia-commons   YFCC100M Flickr CC images (99M images)
  - inaturalist-open-data Wildlife photos
  - met-image            Met Museum bulk public domain
"""

import io
import random
from PIL import Image

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    print("  [aws] boto3 not installed; AWS sources disabled. "
          "Run: pip install boto3")


def _anonymous_client(region='us-east-1'):
    """boto3 S3 client with anonymous access — no credentials needed."""
    return boto3.client('s3',
                        config=Config(signature_version=UNSIGNED),
                        region_name=region)


def stream_s3_bucket(bucket, prefix='', n=200, region='us-east-1',
                     ext=('.jpg', '.jpeg', '.png'), source_label=None,
                     sample_subprefixes=True):
    """Generic streamer for an anonymous S3 bucket.
    Lists keys (sampled with random sub-prefixes for diversity),
    fetches each, decodes as PIL image, yields tuples."""
    if not HAS_BOTO3:
        return

    label = source_label or f"aws_{bucket.replace('-', '_')}"
    print(f"    [aws] listing {bucket}/{prefix} ...")
    keys = []

    try:
        s3 = _anonymous_client(region)

        if sample_subprefixes:
            # Sample random sub-prefixes to get diverse keys without
            # listing the whole bucket (which can be millions of objects)
            sample_chars = list("0123456789abcdef")
            random.shuffle(sample_chars)
            for c in sample_chars[:8]:
                full = prefix + c
                try:
                    paginator = s3.get_paginator('list_objects_v2')
                    for page in paginator.paginate(
                            Bucket=bucket, Prefix=full,
                            PaginationConfig={'MaxItems': n * 2}):
                        for obj in page.get('Contents', []):
                            if obj['Key'].lower().endswith(ext):
                                keys.append(obj['Key'])
                    if len(keys) >= n * 2:
                        break
                except Exception:
                    continue

        # Fallback: list with given prefix only
        if not keys:
            try:
                paginator = s3.get_paginator('list_objects_v2')
                for page in paginator.paginate(
                        Bucket=bucket, Prefix=prefix,
                        PaginationConfig={'MaxItems': n * 3}):
                    for obj in page.get('Contents', []):
                        if obj['Key'].lower().endswith(ext):
                            keys.append(obj['Key'])
                    if len(keys) >= n * 3:
                        break
            except Exception as e:
                print(f"    [aws] list failed for {bucket}: {e}")
                return

    except Exception as e:
        print(f"    [aws] bucket {bucket} access failed: {e}")
        return

    if not keys:
        print(f"    [aws] no matching images in {bucket}/{prefix}")
        return

    random.shuffle(keys)
    print(f"    [aws] {len(keys)} candidates in {bucket}; fetching up to {n}")

    fetched = 0
    for key in keys[:n * 2]:
        if fetched >= n:
            break
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            data = response['Body'].read()
            img = Image.open(io.BytesIO(data)).convert('RGB')
            short = key.split('/')[-1][:40].replace('.', '_')
            yield img, f"{label}_{fetched:05d}_{short}", label
            fetched += 1
        except Exception:
            continue

    print(f"    [aws] {label}: {fetched} images fetched")


# ============================================================
# SPECIFIC BUCKET STREAMERS
# ============================================================

def stream_amazon_bin(n=200):
    """Amazon Bin Image Dataset - 530K product photos in warehouse bins.
    Variety: real product photography, varied lighting, packaging."""
    yield from stream_s3_bucket('aft-vbi-pds', 'bin-images/', n,
                                source_label='aws_amazon_bin')


def stream_open_images(n=200):
    """Google Open Images V7 - 1.7M labeled photographs.
    The standard ML benchmark for object detection."""
    yield from stream_s3_bucket('open-images-dataset', 'train/', n,
                                source_label='aws_open_images',
                                sample_subprefixes=True)


def stream_multimedia_commons(n=200):
    """Yahoo Flickr Creative Commons (YFCC100M) - 99M Flickr photos.
    Massive diversity of real-world photography."""
    yield from stream_s3_bucket('multimedia-commons', 'data/images/', n,
                                source_label='aws_yfcc')


def stream_met_bulk(n=200):
    """Metropolitan Museum bulk download - all public domain artworks."""
    yield from stream_s3_bucket('met-image', '', n,
                                source_label='aws_met_bulk')


def stream_inaturalist_open(n=200):
    """iNaturalist Open Data - wildlife photographs.
    Same source as the API one, but bulk and free at gigabit speeds."""
    yield from stream_s3_bucket('inaturalist-open-data', 'photos/', n,
                                source_label='aws_inat')


# ============================================================
# REGISTRY
# ============================================================

AWS_SOURCES = [
    ("AWS: Amazon Bin Images (530K)",      "no key", stream_amazon_bin),
    ("AWS: Open Images V7 (1.7M)",         "no key", stream_open_images),
    ("AWS: YFCC Multimedia Commons (99M)", "no key", stream_multimedia_commons),
    ("AWS: Met Museum bulk",                "no key", stream_met_bulk),
    ("AWS: iNaturalist Open Data",          "no key", stream_inaturalist_open),
]
