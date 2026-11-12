"""Download eval photos listed in eval/samples.csv from Wikimedia Commons into samples/ (not in git).

uv run python scripts/fetch_samples.py  — fills in license/source_url columns, skips files already present.
"""

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "eval" / "samples.csv"
OUT = ROOT / "samples"
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "GuitarTrace-eval/0.1 (offline research; low volume)"}
WIDTH = 1280  # Commons only serves standard thumbnail widths (https://w.wiki/GHai)


def get(url: str, tries: int = 5) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            time.sleep(10 * (i + 1))
    raise RuntimeError("unreachable")


def info(titles: list[str]) -> dict[str, dict]:
    out = {}
    for k in range(0, len(titles), 40):
        q = {"action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url|extmetadata",
             "iiextmetadatafilter": "LicenseShortName", "iiurlwidth": WIDTH, "titles": "|".join("File:" + t for t in titles[k:k + 40])}
        data = json.loads(get(API + "?" + urllib.parse.urlencode(q)))
        norm = {n["to"]: n["from"] for n in data["query"].get("normalized", [])}
        for p in data["query"]["pages"].values():
            if "imageinfo" in p:
                out[norm.get(p["title"], p["title"]).removeprefix("File:")] = p["imageinfo"][0]
    return out


def main() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    meta = info([r["title"] for r in rows])
    OUT.mkdir(exist_ok=True)
    for r in rows:
        ii = meta.get(r["title"])
        if not ii:
            print(f"{r['name']}: not found on Commons")
            continue
        r["license"] = ii["extmetadata"]["LicenseShortName"]["value"]
        r["source_url"] = ii["descriptionurl"]
        dst = OUT / f"{r['name']}.jpg"
        if not dst.exists():
            try:
                dst.write_bytes(get(ii.get("thumburl") or ii["url"]))
            except urllib.error.HTTPError as e:
                print(f"{r['name']}: {e.code}")
                continue
            print(f"{r['name']}: {dst.stat().st_size // 1024} KB")
            time.sleep(2)
    with CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
