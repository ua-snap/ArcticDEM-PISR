"""Download ArcticDEM mosaic tile assets listed in the tile list. Resumable and idempotent.

For each tile: fetch the STAC catalog -> item JSON (saved as provenance), then each configured asset.
Files stream to *.part and are renamed when complete. A file is skipped if it already exists with the
size reported by the server. Every attempt is recorded in raw/manifest.csv.

Run on a node with internet access (login/transfer node, or `sbatch -p transfer`).
"""
import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests

from arcticdem_pisr.config import Paths, load_config

MANIFEST_FIELDS = ["tile", "asset", "url", "path", "bytes", "last_modified", "etag", "sha256", "status",
                   "message", "time_utc"]
CHUNK = 1 << 20


def get_with_retry(session, url, retries=5, **kw):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=120, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def download_file(session, url, dest, retries=5):
    """Return (status, bytes, last_modified, etag). status in {downloaded, skipped}."""
    head = session.head(url, timeout=60)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    lm, etag = head.headers.get("Last-Modified", ""), head.headers.get("ETag", "").strip('"')
    if dest.exists() and dest.stat().st_size == size:
        return "skipped", size, lm, etag
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for b in r.iter_content(CHUNK):
                        f.write(b)
            if part.stat().st_size != size:
                raise IOError(f"size mismatch {part.stat().st_size} != {size}")
            part.rename(dest)
            return "downloaded", size, lm, etag
        except (requests.RequestException, IOError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def process_tile(tile, catalog_url, assets, raw_dir):
    session = requests.Session()
    rows = []
    tdir = raw_dir / tile
    tdir.mkdir(parents=True, exist_ok=True)
    now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        cat = get_with_retry(session, catalog_url).json()
        item_url = next(l["href"] for l in cat["links"] if l["rel"] == "child")
        item = get_with_retry(session, item_url).json()
        (tdir / "item.json").write_text(json.dumps(item, indent=1))
    except Exception as e:  # noqa: BLE001 - record and move on
        return [dict(tile=tile, asset="item", url=catalog_url, status="failed", message=repr(e), time_utc=now())]
    for asset in assets:
        if asset not in item["assets"]:
            rows.append(dict(tile=tile, asset=asset, status="missing_in_stac", time_utc=now()))
            continue
        url = item["assets"][asset]["href"]
        dest = tdir / url.rsplit("/", 1)[-1]
        try:
            status, size, lm, etag = download_file(session, url, dest)
            rows.append(dict(tile=tile, asset=asset, url=url, path=str(dest), bytes=size, last_modified=lm,
                             etag=etag, sha256=sha256(dest), status=status, time_utc=now()))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(tile=tile, asset=asset, url=url, path=str(dest), status="failed",
                             message=repr(e), time_utc=now()))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tiles", nargs="*", help="subset of tile ids (default: all in tile list)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    paths = Paths(cfg)
    paths.mkdirs()
    adem = cfg["arcticdem"]

    tl = pd.read_csv(paths.tile_list)
    if args.tiles:
        tl = tl[tl.tile.isin(args.tiles)]
    workers = args.workers or adem["download_workers"]
    print(f"Downloading {len(tl)} tiles x {len(adem['assets'])} assets with {workers} workers")

    new = not paths.manifest.exists()
    n_fail = 0
    with open(paths.manifest, "a", newline="") as mf, ThreadPoolExecutor(workers) as ex:
        w = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        futs = {ex.submit(process_tile, r.tile, r.catalog_url, adem["assets"], paths.raw): r.tile
                for r in tl.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            rows = fut.result()
            for row in rows:
                w.writerow(row)
            mf.flush()
            bad = [r for r in rows if r["status"] not in ("downloaded", "skipped")]
            n_fail += len(bad)
            msg = "; ".join(f"{r['asset']}:{r['status']}" for r in bad) or "ok"
            print(f"[{i}/{len(futs)}] {futs[fut]} {msg}", flush=True)
    print(f"Done. {n_fail} failed/missing assets (see {paths.manifest}).")


if __name__ == "__main__":
    main()
