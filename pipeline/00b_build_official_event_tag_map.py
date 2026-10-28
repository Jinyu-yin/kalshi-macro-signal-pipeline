from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import httpx
import pandas as pd

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

from pipeline.utils_paths import output_dir


def _json_dumps_safe(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _fetch_gamma_tags(page_size: int = 10000) -> List[dict]:
    # Gamma /tags returns a list; we keep it simple and fetch once.
    with httpx.Client(timeout=20) as client:
        r = client.get(f"{GAMMA_BASE_URL}/tags", params={"limit": page_size, "offset": 0})
        r.raise_for_status()
        data = r.json()
    return data if isinstance(data, list) else []


def _fetch_gamma_events_by_tag_id(tag_id: str, page_size: int, active: Optional[bool], closed: Optional[bool]) -> List[dict]:
    rows: List[dict] = []
    offset = 0
    params: Dict[str, object] = {"limit": page_size, "offset": offset, "tag_id": tag_id}
    if active is not None:
        params["active"] = str(active).lower()
    if closed is not None:
        params["closed"] = str(closed).lower()

    with httpx.Client(timeout=20) as client:
        while True:
            params["offset"] = offset
            r = client.get(f"{GAMMA_BASE_URL}/events", params=params)
            r.raise_for_status()
            page = r.json()
            if not isinstance(page, list) or not page:
                break
            rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
    return rows


def _derive_category_subcategory_and_tags(tag_labels: List[str]) -> tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """
    Derive:
      category: one of Politics / Markets / Crypto / Economy / finance
      subcategory: first specific label after category (excluding All + category itself)
      tags: remaining labels excluding All
    """
    normalized = [(str(l).strip(), str(l).strip().lower()) for l in tag_labels if l is not None]

    # Keyword-based matching (labels are not guaranteed to be exactly "Crypto"/"finance").
    # We output canonical major category names (Politics/Markets/Crypto/Economy/finance).
    major_keywords = {
        "politics": ["politics", "political"],
        "markets": ["markets", "market"],
        "crypto": [
            "crypto",
            "cryptocurrency",
            "digital-asset",
            "digital asset",
            "digital-assets",
            "digital assets",
            "digital-asset",
            "digital asset",
        ],
        "economy": ["economy", "economics", "economic"],
        "finance": ["finance", "financial"],
    }
    canonical_major = {
        "politics": "Politics",
        "markets": "Markets",
        "crypto": "Crypto",
        "economy": "Economy",
        "finance": "finance",
    }

    def _matches_major(low: str, major_key: str) -> bool:
        kws = major_keywords.get(major_key, [])
        return any(k in low for k in kws)

    # choose category key by priority order
    cat_key: Optional[str] = None
    for major_key in ["politics", "markets", "crypto", "economy", "finance"]:
        for original, low in normalized:
            if original.lower() == "all":
                continue
            if _matches_major(low, major_key):
                cat_key = major_key
                break
        if cat_key is not None:
            break

    cat: Optional[str] = canonical_major.get(cat_key) if cat_key is not None else None

    # tags array: keep all labels except "All" and (optionally) the major category label itself
    kept_tags: List[str] = []
    for original, low in normalized:
        if original.lower() == "all":
            continue
        if cat_key is not None and _matches_major(low, cat_key):
            continue
        kept_tags.append(original)

    subcat: Optional[str] = None
    # choose first non-generic label after category
    for t in kept_tags:
        if not t:
            continue
        subcat = t
        break

    if not kept_tags:
        return cat, subcat, None
    return cat, subcat, kept_tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache official event-tag mapping for Polymarket markets")
    parser.add_argument(
        "--tag-slugs",
        type=str,
        default="fed-rates,macro-indicators,imf,government-shutdown,libor,nyse",
        help="Comma-separated tag slugs. Use 'all' to fetch full /events (slow).",
    )
    parser.add_argument(
        "--tag-ids",
        type=str,
        default="",
        help="Comma-separated Gamma tag ids (numeric). If provided, fetch events by these tag ids directly.",
    )
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--active", type=str, default="none", choices=["none", "true", "false"])
    parser.add_argument("--closed", type=str, default="none", choices=["none", "true", "false"])
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache file")
    args = parser.parse_args()

    out_dir = output_dir() / "markets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "official_event_tag_map.csv"
    if out_path.exists() and not args.force:
        print(f"[00b] cache exists, skip: {out_path}")
        return

    active: Optional[bool] = None
    closed: Optional[bool] = None
    if args.active != "none":
        active = args.active.lower() == "true"
    if args.closed != "none":
        closed = args.closed.lower() == "true"

    tag_slugs_arg = (args.tag_slugs or "").strip()
    if not tag_slugs_arg:
        tag_slugs_arg = "fed-rates,macro-indicators"

    tag_ids_arg = (args.tag_ids or "").strip()
    tag_ids_seed: List[str] = []
    if tag_ids_arg:
        tag_ids_seed = [s.strip() for s in tag_ids_arg.split(",") if s.strip()]

    if tag_slugs_arg.lower() == "all" and not tag_ids_seed:
        # Fallback: slow full scan (kept for completeness).
        print("[00b] tag-slugs=all -> full /events scan (slow).")
        with httpx.Client(timeout=20) as client:
            offset = 0
            page_size = args.page_size
            rows: List[dict] = []
            while True:
                params: Dict[str, object] = {"limit": page_size, "offset": offset}
                if active is not None:
                    params["active"] = str(active).lower()
                if closed is not None:
                    params["closed"] = str(closed).lower()
                r = client.get(f"{GAMMA_BASE_URL}/events", params=params)
                r.raise_for_status()
                page = r.json()
                if not isinstance(page, list) or not page:
                    break
                rows.extend(page)
                if len(page) < page_size:
                    break
                offset += page_size
        events = rows
    else:
        # Merge: tag_ids_seed + resolved tag-slugs ids (if any)
        tag_slugs = [s.strip() for s in tag_slugs_arg.split(",") if s.strip() and s.strip().lower() != "all"]

        need_ids: Set[str] = set(tag_ids_seed)

        if tag_slugs:
            print(f"[00b] fetching /tags (resolve ids for {len(tag_slugs)} tag slugs)...")
            tags = _fetch_gamma_tags(page_size=20000)
            tag_slug_to_id = {
                str(t.get("slug", "")).strip(): str(t.get("id", "")).strip()
                for t in tags
                if t.get("slug") and t.get("id") is not None
            }

            for s in tag_slugs:
                tid = tag_slug_to_id.get(s)
                if tid:
                    need_ids.add(str(tid))
                else:
                    print(f"[00b] WARN cannot find tag id for slug={s}")

        if not need_ids:
            raise RuntimeError("No tag ids resolved; provide --tag-ids or a valid --tag-slugs list.")

        need_ids_sorted = sorted(need_ids)
        print(f"[00b] total tag_ids to fetch: {len(need_ids_sorted)}")

        events: List[dict] = []
        for tid in need_ids_sorted:
            print(f"[00b] fetching events by tag_id={tid} ...")
            evs = _fetch_gamma_events_by_tag_id(tid, page_size=args.page_size, active=active, closed=closed)
            events.extend(evs)
            print(f"[00b] tag_id={tid} events={len(evs):,}")

    # Build mapping: market_id -> union of official tags
    market_to_event: Dict[str, dict] = {}
    market_to_tag_slugs: Dict[str, Set[str]] = {}
    market_to_tag_labels: Dict[str, Set[str]] = {}
    market_to_tag_dicts: Dict[str, Dict[str, dict]] = {}

    for ev in events:
        if not isinstance(ev, dict):
            continue
        ev_id = str(ev.get("id", "")).strip()
        if not ev_id:
            continue
        ev_title = ev.get("title")
        ev_slug = ev.get("slug")
        tags = ev.get("tags", [])
        tag_dicts_by_slug: Dict[str, dict] = {}
        tag_slugs: List[str] = []
        tag_labels: List[str] = []
        if isinstance(tags, list):
            for t in tags:
                if not isinstance(t, dict):
                    continue
                slug = str(t.get("slug", "")).strip()
                label = str(t.get("label", "")).strip()
                if slug:
                    tag_dicts_by_slug[slug] = t
                    tag_slugs.append(slug)
                if label:
                    tag_labels.append(label)

        markets = ev.get("markets", [])
        if not isinstance(markets, list):
            continue
        for m in markets:
            if not isinstance(m, dict):
                continue
            mid = str(m.get("id", "")).strip()
            if not mid:
                continue

            if mid not in market_to_event:
                market_to_event[mid] = {
                    "market_id": mid,
                    "event_id": ev_id,
                    "event_title": ev_title,
                    "event_slug": ev_slug,
                }

            ms = market_to_tag_slugs.setdefault(mid, set())
            ms.update(tag_slugs)
            ml = market_to_tag_labels.setdefault(mid, set())
            ml.update(tag_labels)

            md = market_to_tag_dicts.setdefault(mid, {})
            for slug, td in tag_dicts_by_slug.items():
                md[slug] = td

    rows_out: List[dict] = []
    for mid, ev_meta in market_to_event.items():
        slugs_sorted = sorted(market_to_tag_slugs.get(mid, set()))
        labels_sorted = sorted(market_to_tag_labels.get(mid, set()))
        td = market_to_tag_dicts.get(mid, {})
        tags_list_sorted = [td[k] for k in sorted(td.keys())]

        # derive category/subcategory/tags from label strings
        event_category, event_subcategory, event_tags = _derive_category_subcategory_and_tags(labels_sorted)

        rows_out.append(
            {
                "market_id": mid,
                "event_id": str(ev_meta.get("event_id", "")).strip(),
                "event_title": ev_meta.get("event_title"),
                "event_slug": ev_meta.get("event_slug"),
                "official_tag_slugs": _json_dumps_safe(slugs_sorted),
                "official_tag_labels": _json_dumps_safe(labels_sorted),
                "official_tags": _json_dumps_safe(tags_list_sorted),
                "category": event_category,
                "subcategory": event_subcategory,
                "tags": _json_dumps_safe(event_tags),
            }
        )

    df = pd.DataFrame(rows_out)
    df = df.drop_duplicates(subset=["market_id"], keep="first")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    tagged_markets = int(df["official_tag_slugs"].notna().sum()) if "official_tag_slugs" in df.columns else 0
    print(f"[00b] saved: {out_path}")
    print(f"[00b] rows={len(df):,} tagged_markets={tagged_markets:,}")


if __name__ == "__main__":
    main()

