"""Parallel bulk download of a native OLMo-core checkpoint from gs://ai2-llm to local disk.

Loading a DCP (distributed) checkpoint directly from GCS is very slow: the 512->N
reshard issues many scattered small range reads. Downloading every shard once, in
parallel, to local NVMe is far faster and is reused across anneal runs.

The public ai2-llm bucket is readable anonymously (no credentials needed).

Usage:
    python download_checkpoint.py \
        --gs gs://ai2-llm/checkpoints/OLMo3-7B-swafix/step6000 \
        --out $OUT_DIR/native/swafix-step6000 \
        --workers 64
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud.storage import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gs", required=True, help="gs://bucket/prefix of the checkpoint dir.")
    p.add_argument("--out", required=True, help="Local output dir.")
    p.add_argument("--workers", type=int, default=64, help="Parallel download threads (default 64).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.gs.startswith("gs://"):
        raise ValueError(f"--gs must start with gs:// (got {args.gs})")

    bucket_name, prefix = args.gs[len("gs://"):].split("/", 1)
    prefix = prefix.rstrip("/") + "/"

    client = Client.create_anonymous_client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    total_bytes = sum(b.size for b in blobs)
    if not blobs:
        raise RuntimeError(f"No objects found at {args.gs}")
    log.info("%d files, %.1f GB -> %s", len(blobs), total_bytes / 1e9, args.out)

    def fetch(blob) -> int:
        rel = blob.name[len(prefix):]
        if not rel:
            return 0
        dst = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # Resume: skip files already fully downloaded.
        if os.path.exists(dst) and os.path.getsize(dst) == blob.size:
            return blob.size
        blob.download_to_filename(dst)
        return blob.size

    done = done_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch, b): b for b in blobs}
        for fut in as_completed(futures):
            done += 1
            done_bytes += fut.result()
            if done % 200 == 0 or done == len(blobs):
                log.info("  %d/%d files, %.1f/%.1f GB", done, len(blobs), done_bytes / 1e9, total_bytes / 1e9)

    log.info("DONE -> %s", args.out)


if __name__ == "__main__":
    main()
