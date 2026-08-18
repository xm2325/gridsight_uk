from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "_site" / "index.html"
BUILD_ID = "v4.11-primary-16-gallery"


def main() -> None:
    if not INDEX.exists():
        raise FileNotFoundError("Build v4.6 Pages, v4.8 dataset gallery, and v4.9 cases first")

    page = INDEX.read_text(encoding="utf-8")

    if '<section id="gallery">' not in page:
        raise RuntimeError("Original three-image gallery section not found")
    if '<section id="dataset">' not in page:
        raise RuntimeError("Full 16-image dataset gallery section not found")
    dataset_cards_before = page.count('class="dataset-card"')
    if dataset_cards_before != 16:
        raise RuntimeError(f"Expected 16 dataset cards before promotion, found {dataset_cards_before}")

    # Remove the old three-card showcase section entirely. Those same three sources remain
    # represented inside the full provenance gallery, with their evidence roles preserved.
    gallery_pattern = re.compile(
        r'<section id="gallery">.*?</section>\s*(?=<section id="reference-details">)',
        flags=re.DOTALL,
    )
    page, removed = gallery_pattern.subn("", page, count=1)
    if removed != 1:
        raise RuntimeError(f"Expected to remove one three-image gallery, removed {removed}")

    # Promote the all-source dataset section to the canonical gallery anchor.
    page = page.replace('<section id="dataset">', '<section id="gallery" data-gallery-count="16">', 1)
    page = page.replace(
        '<div class="eyebrow">Dataset gallery · source provenance</div>',
        '<div class="eyebrow">Full source gallery · all 16 images</div>',
        1,
    )
    page = page.replace(
        '<h2>All 16 source images used or formally reviewed in this repository</h2>',
        '<h2>All 16 project source images — with split role, provenance and case study</h2>',
        1,
    )
    page = page.replace(
        'Every unique source currently represented by the repository manifests is shown below. Roles are kept separate:',
        'This is the primary image gallery: every unique source currently represented by the repository manifests is shown below. Roles are kept separate:',
        1,
    )

    # Keep one unambiguous gallery item in the sticky navigation.
    page = page.replace('<a href="#gallery">Tower gallery</a>', '<a href="#gallery">Gallery · 16</a>', 1)
    page = page.replace('<a href="#dataset">Dataset</a>', '', 1)

    # The old hero copy referred to the previous three-image-only presentation.
    page = page.replace(
        'The page now includes three tower configurations rather than one.',
        'The page includes a 16-source gallery covering training, validation, holdout, stress and material-development roles.',
        1,
    )

    # Add a machine-readable and human-visible build marker so the deployed Pages version
    # can be distinguished from a stale CDN/browser copy without guessing.
    if '<head>' not in page or '</footer>' not in page:
        raise RuntimeError("Could not locate head/footer for build marker")
    page = page.replace('<head>', f'<head>\n<meta name="gridsight-build" content="{BUILD_ID}">', 1)
    page = page.replace(
        '</footer>',
        f'<br><span class="small">Build · {BUILD_ID} · primary gallery 16/16</span></footer>',
        1,
    )

    # Guardrails against accidentally shipping the confusing old presentation again.
    if 'Three source images, three tower configurations' in page:
        raise RuntimeError("Old three-image gallery heading still present")
    if 'Tower gallery' in page:
        raise RuntimeError("Old Tower gallery navigation label still present")
    if '<section id="dataset">' in page:
        raise RuntimeError("Dataset section was not promoted to the gallery anchor")
    if page.count('<section id="gallery"') != 1:
        raise RuntimeError("Expected exactly one canonical gallery section")
    if page.count('class="dataset-card"') != 16:
        raise RuntimeError("Primary gallery does not contain all 16 source cards")
    if page.count('class="case-open"') != 16:
        raise RuntimeError("Primary gallery lost one or more case-study controls")
    if page.count(BUILD_ID) != 2:
        raise RuntimeError("Expected build marker in meta tag and visible footer")

    INDEX.write_text(page, encoding="utf-8")
    print({
        "build_id": BUILD_ID,
        "primary_gallery_cards": page.count('class="dataset-card"'),
        "case_buttons": page.count('class="case-open"'),
        "old_three_image_gallery_removed": True,
        "gallery_anchor_count": page.count('<section id="gallery"'),
        "index_bytes": INDEX.stat().st_size,
    })


if __name__ == "__main__":
    main()
