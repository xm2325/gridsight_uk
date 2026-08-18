from __future__ import annotations

import hashlib
import html
import json
import shutil
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"
ASSETS = SITE / "assets"
DATASET = ASSETS / "dataset"
THUMBS = DATASET / "thumbs"
FULL = DATASET / "full"
UA = "GridSight-UK-v4.8-pages-gallery/1.0 (+https://github.com/xm2325/gridsight_uk)"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fetch(url: str) -> bytes:
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "image/jpeg,image/*;q=0.8,*/*;q=0.1"})
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read()
        except Exception as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {url}: {last}")


def copy_or_exact_original(src: Path, dst: Path, original_url: str | None = None, expected_sha256: str | None = None) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    status = "project_runtime_source"
    exact = False
    error = None
    if original_url and expected_sha256:
        try:
            payload = fetch(original_url)
            digest = hashlib.sha256(payload).hexdigest()
            if digest != expected_sha256:
                raise RuntimeError(f"SHA-256 mismatch: {digest} != {expected_sha256}")
            if not payload.startswith(b"\xff\xd8"):
                raise RuntimeError("verified source bytes are not JPEG")
            dst.write_bytes(payload)
            status = "verified_commons_original"
            exact = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if not exact:
        shutil.copy2(src, dst)
    with Image.open(dst) as im:
        w, h = im.size
    return {
        "width": w,
        "height": h,
        "bytes": dst.stat().st_size,
        "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        "display_source_kind": status,
        "canonical_original_exact": exact,
        "original_fetch_error": error,
    }


def make_thumb(src: Path, dst: Path, max_size=(560, 420)) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        thumb = ImageOps.contain(im.convert("RGB"), max_size, Image.Resampling.LANCZOS)
        thumb.save(dst, "WEBP", quality=90, method=6)
        return thumb.size


def badge(text: str, cls: str = "") -> str:
    return f'<span class="dataset-badge {cls}">{html.escape(text)}</span>'


def card(entry: dict) -> str:
    role_badges = "".join(badge(x, "role") for x in entry["roles"])
    source_badge = badge(entry["source_family"], "source")
    split_badge = badge(entry["split"], "split") if entry.get("split") else ""
    quality_label = "verified original" if entry["canonical_original_exact"] else "project source pixels"
    quality_badge = badge(quality_label, "quality")
    commons = ""
    if entry.get("commons_url"):
        commons = f'<a href="{html.escape(entry["commons_url"])}" target="_blank" rel="noopener">Commons ↗</a>'
    return f'''
    <article class="dataset-card" data-filter="{html.escape(entry['filter_group'])}">
      <button class="dataset-image-button" type="button" data-full="{html.escape(entry['full_path'])}" aria-label="Open full local image for {html.escape(entry['record_id'])}">
        <img src="{html.escape(entry['thumb_path'])}" width="{entry['thumb_width']}" height="{entry['thumb_height']}" loading="lazy" decoding="async" alt="{html.escape(entry['record_id'])} transmission tower source image">
      </button>
      <div class="dataset-card-body">
        <div class="dataset-card-title"><b>{html.escape(entry['record_id'])}</b><span>{entry['width']}×{entry['height']}</span></div>
        <div class="dataset-badges">{source_badge}{split_badge}{quality_badge}{role_badges}</div>
        <p>{html.escape(entry['description'])}</p>
        <div class="dataset-hash">Displayed SHA-256 · {html.escape(entry['sha256'][:16])}…</div>
      </div>
      <div class="dataset-links">
        <a href="{html.escape(entry['full_path'])}" target="_blank" rel="noopener">Full local image ↗</a>
        <a href="{html.escape(entry['source_page'])}" target="_blank" rel="noopener">Geograph source ↗</a>
        {commons}
      </div>
    </article>'''


def main() -> None:
    index_path = SITE / "index.html"
    if not index_path.exists():
        raise FileNotFoundError("Build the main Pages site before adding the dataset gallery")

    legacy_manifest = load_json(ROOT / "data/image_sources.json")
    legacy_runtime = load_json(ROOT / "reports/v2_3_frozen_image_download.json")
    v29 = load_json(ROOT / "reports/v2_9_unseen_candidate_manifest.json")
    v44 = load_json(ROOT / "reports/v4_4_angle_material_candidate_manifest.json")

    runtime_by_id = {r["record_id"]: r for r in legacy_runtime["images"]}
    v29_by_id = {r["record_id"]: r for r in v29["candidates"]}
    v44_by_id = {r["record_id"]: r for r in v44["candidates"]}

    role_map = {
        "POS_1283842": (["original training"], "legacy-train", "Original component-training source."),
        "POS_190181": (["original training"], "legacy-train", "Original component-training source."),
        "POS_7060068": (["original training"], "legacy-train", "Original component-training source."),
        "POS_3778704": (["original training"], "legacy-train", "Original component-training source."),
        "POS_291727": (["original training"], "legacy-train", "Original component-training source."),
        "POS_5442616": (["original validation"], "legacy-val", "Original source-separated validation image."),
        "POS_2326530": (["original test", "v2 showcase"], "legacy-test", "Predeclared source-separated test/showcase image used in early component-model evaluation."),
        "POS_3437435": (["v3.8 primary final holdout"], "holdout", "Primary frozen final component holdout; retired after headline evaluation."),
        "POS_7561805": (["v3.8 scale final holdout"], "holdout", "Secondary frozen final holdout selected to test scale generalisation; retired after evaluation."),
        "POS_6610209": (["v4 morphology training"], "v4-dev", "Close-range, non-standard geometry source accepted as a new morphology training stratum."),
        "POS_8091164": (["v4 validation"], "v4-val", "Distinct compact/angle tower used as the fixed v4 validation source."),
        "POS_7072688": (["scale stress"], "stress", "Long-range source retained for scale-stress review; component pixels were too small for reliable manual boxes."),
        "POS_7478407": (["extreme scale stress"], "stress", "Extreme long-range deployment-scale stress source; not used as component ground truth."),
        "POS_8090535": (["material reference", "angle / strain"], "material", "Source-assisted material reference and angle/strain development source."),
        "POS_2952166": (["material adaptive development", "main showcase"], "material", "Mixed-material source used in the current adaptive SigLIP2 material-development showcase."),
        "POS_7630781": (["v4 frozen detection holdout", "terminal morphology"], "holdout", "Terminal-tower source frozen before v4 detection evaluation; not a material prediction source."),
    }

    shutil.rmtree(DATASET, ignore_errors=True)
    THUMBS.mkdir(parents=True, exist_ok=True)
    FULL.mkdir(parents=True, exist_ok=True)
    entries = []

    # Legacy model runtime remains unchanged. The portfolio gallery separately tries the exact
    # Commons bytes from the frozen manifest, falling back to the project runtime derivative.
    for item in legacy_manifest["images"]:
        rid = item["record_id"]
        runtime = runtime_by_id[rid]
        runtime_src = ROOT / runtime["path"]
        roles, group, description = role_map[rid]
        full_name = f"{rid}.jpg"
        thumb_name = f"{rid}.webp"
        full_path = FULL / full_name
        info = copy_or_exact_original(runtime_src, full_path, item["url"], item["expected_sha256"])
        tw, th = make_thumb(full_path, THUMBS / thumb_name)
        commons_name = quote(item["commons_file_name"], safe="_-.()")
        entries.append({
            "record_id": rid,
            "source_family": "Geograph / Commons legacy",
            "split": item["split"],
            "roles": roles,
            "filter_group": group,
            "description": description,
            **info,
            "runtime_dimensions": [runtime["width_px"], runtime["height_px"]],
            "source_page": runtime.get("geograph_page_url") or f"https://www.geograph.org.uk/photo/{rid.removeprefix('POS_')}",
            "commons_url": f"https://commons.wikimedia.org/wiki/File:{commons_name}",
            "full_path": f"assets/dataset/full/{full_name}",
            "thumb_path": f"assets/dataset/thumbs/{thumb_name}",
            "thumb_width": tw,
            "thumb_height": th,
        })

    for rid in ["POS_3437435", "POS_7561805", "POS_6610209", "POS_8091164", "POS_7072688", "POS_7478407"]:
        item = v29_by_id[rid]
        src = ROOT / item["path"]
        roles, group, description = role_map[rid]
        full_name = f"{rid}.jpg"
        thumb_name = f"{rid}.webp"
        full_path = FULL / full_name
        info = copy_or_exact_original(src, full_path)
        tw, th = make_thumb(full_path, THUMBS / thumb_name)
        entries.append({
            "record_id": rid,
            "source_family": "Geograph v2.9/v3",
            "split": "holdout" if group == "holdout" else "development",
            "roles": roles,
            "filter_group": group,
            "description": description,
            **info,
            "source_page": item["photo_page_url"],
            "commons_url": None,
            "full_path": f"assets/dataset/full/{full_name}",
            "thumb_path": f"assets/dataset/thumbs/{thumb_name}",
            "thumb_width": tw,
            "thumb_height": th,
        })

    for rid in ["POS_8090535", "POS_2952166", "POS_7630781"]:
        item = v44_by_id[rid]
        src = ROOT / item["path"]
        roles, group, description = role_map[rid]
        full_name = f"{rid}.jpg"
        thumb_name = f"{rid}.webp"
        full_path = FULL / full_name
        info = copy_or_exact_original(src, full_path)
        tw, th = make_thumb(full_path, THUMBS / thumb_name)
        entries.append({
            "record_id": rid,
            "source_family": "Geograph v4.4",
            "split": "holdout" if rid == "POS_7630781" else "development",
            "roles": roles,
            "filter_group": group,
            "description": description,
            **info,
            "source_page": item["page_url"],
            "commons_url": None,
            "full_path": f"assets/dataset/full/{full_name}",
            "thumb_path": f"assets/dataset/thumbs/{thumb_name}",
            "thumb_width": tw,
            "thumb_height": th,
        })

    ids = [e["record_id"] for e in entries]
    if len(entries) != 16 or len(set(ids)) != 16:
        raise RuntimeError(f"Expected 16 unique dataset images, got {len(entries)} / {len(set(ids))} unique")

    groups = [
        ("Legacy component benchmark", [e for e in entries if e["filter_group"].startswith("legacy-")]),
        ("Morphology, stress and v3 holdouts", [e for e in entries if e["record_id"] in {"POS_3437435","POS_7561805","POS_6610209","POS_8091164","POS_7072688","POS_7478407"}]),
        ("v4.4 angle / material / terminal cycle", [e for e in entries if e["record_id"] in {"POS_8090535","POS_2952166","POS_7630781"}]),
    ]
    group_html = []
    for title, rows in groups:
        group_html.append(
            f'<div class="dataset-group"><div class="dataset-group-head"><h3>{html.escape(title)}</h3><span>{len(rows)} source images</span></div>'
            f'<div class="dataset-grid">{"".join(card(e) for e in rows)}</div></div>'
        )

    exact_legacy = sum(1 for e in entries if e["source_family"].startswith("Geograph / Commons") and e["canonical_original_exact"])
    section = f'''
<section id="dataset">
  <div class="eyebrow">Dataset gallery · source provenance</div>
  <h2>All 16 source images used or formally reviewed in this repository</h2>
  <p class="muted dataset-intro">Every unique source currently represented by the repository manifests is shown below. Roles are kept separate: training/validation, frozen holdout, material development, and scale-stress review are not interchangeable evidence. For the seven legacy records, the gallery tries the frozen Wikimedia Commons original bytes first; model runtime hydration remains unchanged.</p>
  <div class="dataset-summary">
    <div class="metric good"><b>16</b><span>unique source images shown</span></div>
    <div class="metric"><b>7</b><span>legacy train / validation / test sources</span></div>
    <div class="metric warn"><b>3</b><span>frozen holdout sources across v3/v4</span></div>
    <div class="metric"><b>{exact_legacy}/7</b><span>legacy cards using verified Commons original bytes in this build</span></div>
  </div>
  <div class="dataset-filters" role="group" aria-label="Filter source-image gallery">
    <button class="dataset-filter active" data-show="all">All 16</button>
    <button class="dataset-filter" data-show="legacy">Legacy benchmark</button>
    <button class="dataset-filter" data-show="v4">v4 development</button>
    <button class="dataset-filter" data-show="holdout">Frozen holdouts</button>
    <button class="dataset-filter" data-show="stress">Scale stress</button>
  </div>
  {''.join(group_html)}
  <div class="panel provenance-panel">
    <h3>Data provenance contract</h3>
    <div class="provenance-grid">
      <span>✓ source record ID retained</span>
      <span>✓ original Geograph source page linked</span>
      <span>✓ exact Commons originals verified by SHA-256 when available</span>
      <span>✓ displayed local image SHA-256 recorded</span>
      <span>✓ train / validation / holdout / stress role stated</span>
      <span>✓ third-party source images are not presented as project-owned assets</span>
    </div>
    <p class="muted small">The gallery is a provenance and portfolio view. Candidate-only and assistant-provisional labels are not independent engineering inspection ground truth. A “verified original” badge refers to image-byte identity, not label quality.</p>
  </div>
</section>
'''

    css = r'''
.dataset-intro{max-width:960px}.dataset-summary{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}.dataset-filters{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.dataset-filter{appearance:none;border:1px solid var(--line);border-radius:999px;background:#0d1722;color:var(--muted);padding:7px 11px;cursor:pointer}.dataset-filter.active{border-color:var(--green);color:var(--green);background:rgba(86,208,164,.08)}.dataset-group{margin-top:24px}.dataset-group-head{display:flex;align-items:end;justify-content:space-between;gap:12px;margin-bottom:10px}.dataset-group-head span{color:var(--muted);font-size:12px}.dataset-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:12px}.dataset-card{display:flex;flex-direction:column;min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:15px;overflow:hidden}.dataset-card[hidden]{display:none}.dataset-image-button{display:block;width:100%;border:0;padding:0;background:#050a0f;cursor:zoom-in}.dataset-image-button img{display:block;width:100%;height:250px;object-fit:contain;background:#050a0f}.dataset-card-body{padding:11px 12px 10px}.dataset-card-title{display:flex;justify-content:space-between;gap:10px;align-items:baseline}.dataset-card-title span{font-size:11px;color:var(--muted)}.dataset-badges{display:flex;gap:5px;flex-wrap:wrap;margin:7px 0 8px}.dataset-badge{display:inline-flex;border:1px solid var(--line);border-radius:999px;padding:2px 6px;font-size:10px;color:var(--muted)}.dataset-badge.role{color:var(--green)}.dataset-badge.split{color:var(--amber)}.dataset-badge.source{color:var(--blue)}.dataset-badge.quality{color:#d9e6ef}.dataset-card p{margin:0;color:var(--muted);font-size:12px;min-height:57px}.dataset-hash{margin-top:9px;color:#73889a;font:10px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace}.dataset-links{display:flex;gap:8px;flex-wrap:wrap;margin-top:auto;border-top:1px solid var(--line);padding:9px 12px}.dataset-links a{font-size:11px;color:var(--muted);text-decoration:none}.dataset-links a:hover{color:var(--green)}.provenance-panel{margin-top:18px}.provenance-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px 18px;color:var(--muted);font-size:13px}@media(max-width:850px){.dataset-summary{grid-template-columns:1fr 1fr}.provenance-grid{grid-template-columns:1fr}}@media(max-width:520px){.dataset-summary{grid-template-columns:1fr}.dataset-image-button img{height:220px}}
'''

    js = r'''
<script>
(function(){
  const filters=document.querySelectorAll('.dataset-filter');
  const cards=document.querySelectorAll('.dataset-card');
  const groups=document.querySelectorAll('.dataset-group');
  function category(card,show){
    const f=card.dataset.filter||'';
    if(show==='all') return true;
    if(show==='legacy') return f.startsWith('legacy-');
    if(show==='holdout') return f==='holdout';
    if(show==='stress') return f==='stress';
    if(show==='v4') return ['v4-dev','v4-val','material','holdout'].includes(f);
    return true;
  }
  filters.forEach(btn=>btn.addEventListener('click',()=>{
    filters.forEach(x=>x.classList.toggle('active',x===btn));
    const show=btn.dataset.show;
    cards.forEach(card=>card.hidden=!category(card,show));
    groups.forEach(group=>{group.hidden=![...group.querySelectorAll('.dataset-card')].some(x=>!x.hidden);});
  }));
  document.querySelectorAll('.dataset-image-button').forEach(btn=>btn.addEventListener('click',()=>{
    const modal=document.getElementById('modal');
    const image=document.getElementById('modal-img');
    if(modal && image){image.src=btn.dataset.full;image.alt='Expanded source image';modal.classList.add('open');modal.setAttribute('aria-hidden','false');}
    else{window.open(btn.dataset.full,'_blank');}
  }));
})();
</script>
'''

    page = index_path.read_text(encoding="utf-8")
    if 'id="dataset"' in page:
        raise RuntimeError("Dataset gallery already present")
    if '<section id="evidence">' not in page:
        raise RuntimeError("Could not locate evidence section insertion point")
    page = page.replace('<section id="evidence">', section + '\n<section id="evidence">', 1)
    page = page.replace('<a href="#evidence">Evidence</a>', '<a href="#dataset">Dataset</a><a href="#evidence">Evidence</a>', 1)
    page = page.replace('</style>', css + '\n</style>', 1)
    page = page.replace('</body>', js + '\n</body>', 1)
    index_path.write_text(page, encoding="utf-8")

    manifest = {
        "version": "v4.8-full-dataset-gallery-hires",
        "unique_source_images": len(entries),
        "legacy_verified_commons_originals": exact_legacy,
        "entries": entries,
        "claim_boundary": "Portfolio provenance gallery only; source role and evidence status must not be collapsed into a single model-performance claim.",
    }
    (DATASET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total_full = sum((FULL / f"{rid}.jpg").stat().st_size for rid in ids)
    if total_full > 120_000_000:
        raise RuntimeError(f"Dataset gallery full-image payload unexpectedly large: {total_full}")
    print(json.dumps({
        "index_bytes": index_path.stat().st_size,
        "unique_source_images": len(entries),
        "legacy_verified_commons_originals": exact_legacy,
        "thumbnail_bytes": sum((THUMBS / f"{rid}.webp").stat().st_size for rid in ids),
        "full_image_bytes": total_full,
        "manifest": str((DATASET / 'manifest.json').relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
