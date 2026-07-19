#!/usr/bin/env python3
"""Validate the preserved seven-section RSQ manuscript and arXiv archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MAIN = PAPER / "main.tex"
BIB = PAPER / "refs.bib"
BBL = PAPER / "main.bbl"
POPULAR = PAPER / "prx-popular-summary.tex"
PDF = PAPER / "main.pdf"
ARCHIVE = PAPER / "arxiv-source-rsq.zip"
MANIFEST = PAPER / "tables" / "amortized_asset_manifest.json"

EXPECTED_BIB_SHA256 = (
    "473d187cdc6398b6bdf001110da5d64f280dacb92834a99bcb5c5b1f8236226d"
)
EXPECTED_MANIFEST_SHA256 = (
    "b38b93c768baa3f5eeb685482f6a9263621e3367b26afd6f2467404050280407"
)
EXPECTED_TITLE = "Task-Weighted Observable Subspaces for Repeated Multi-Angle QAOA"
EXPECTED_SECTIONS = (
    "Introduction",
    "Background",
    "Related Work",
    "Shortcomings of Static and Unweighted Subspaces",
    "Task-Weighted Observable Subspaces",
    "Experiments",
    "Conclusion",
)
EXPECTED_SUBSECTIONS = (0, 3, 0, 3, 4, 3, 0)
EXPECTED_FIGURES = (
    "figures/figure_amortized_protocol.pdf",
    "figures/figure_amortized_stream.pdf",
    "figures/figure_amortized_exact_audit.pdf",
    "figures/figure1_tradeoff.pdf",
    "figures/figure_amortized_shot_audit.pdf",
    "figures/figure2_family.pdf",
    "figures/figure3_operator_budget.pdf",
    "figures/figure4_rank.pdf",
    "figures/figure9_refresh_frequency.pdf",
    "figures/figure10_dimension_by_depth.pdf",
    "figures/figure11_quality_cost_frontier.pdf",
    "figures/figure12_family_effects.pdf",
    "figures/figure13_reproducibility_map.pdf",
)
EXPECTED_TABLES = (
    "tables/table1_summary.tex",
    "tables/table_amortized_exact_audit.tex",
    "tables/table_development_gate.tex",
    "tables/table_protocol.tex",
    "tables/table_amortized_shot_audit.tex",
)
FORBIDDEN_TEXT = (
    "acknowledg",
    "impact" + " statement",
    "proof" + " sketch",
    "2607." + "06758",
    "smith2026" + "adaptive",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_comments(source: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", source)


def normalized_title(source: str) -> str:
    match = re.search(r"\\title\{(.*?)\}\s*\\author", source, flags=re.S)
    if not match:
        return ""
    title = match.group(1).replace(r"\textbf{", "")
    title = title.replace("\\\\", " ").replace("{", "").replace("}", "")
    return " ".join(title.split())


def word_count(source: str) -> int:
    source = re.sub(
        r"\\(?:cite|ref|eqref|label)\w*(?:\[[^\]]*\])?\{[^{}]*\}",
        " ",
        source,
    )
    source = re.sub(r"\\[A-Za-z@]+(?:\[[^\]]*\])?", " ", source)
    source = re.sub(r"[{}$\\_^~]", " ", source)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", source))


def included_paths(source: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    clean = without_comments(source)
    figures = tuple(
        re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", clean)
    )
    tables = tuple(re.findall(r"\\input\{([^{}]+)\}", clean))
    return figures, tables


def archive_sources() -> dict[str, Path]:
    sources = {
        "main.tex": MAIN,
        "refs.bib": BIB,
        "main.bbl": BBL,
        "prx-popular-summary.tex": POPULAR,
    }
    sources.update({relative: PAPER / relative for relative in EXPECTED_FIGURES})
    sources.update({relative: PAPER / relative for relative in EXPECTED_TABLES})
    require(len(sources) == 22, "internal archive inventory is not 22 files")
    return sources


def structural_report(source: str) -> dict[str, object]:
    split = source.find(r"\appendix")
    require(split >= 0, "appendix transition is missing")
    main = source[:split]
    appendix = source[split:]
    section_matches = list(re.finditer(r"^\\section\{([^{}]+)\}", main, flags=re.M))
    sections = tuple(match.group(1) for match in section_matches)
    require(sections == EXPECTED_SECTIONS, f"main sections differ: {sections}")

    subsection_counts: list[int] = []
    for index, match in enumerate(section_matches):
        end = (
            section_matches[index + 1].start()
            if index + 1 < len(section_matches)
            else len(main)
        )
        subsection_counts.append(
            len(re.findall(r"^\\subsection\{", main[match.start():end], flags=re.M))
        )
    require(
        tuple(subsection_counts) == EXPECTED_SUBSECTIONS,
        f"main subsection counts differ: {subsection_counts}",
    )

    figures, tables = included_paths(source)
    require(figures == EXPECTED_FIGURES, f"figure inputs differ: {figures}")
    require(tables == EXPECTED_TABLES, f"table inputs differ: {tables}")
    require(len(set(figures)) == 13, "figure inputs are not 13 unique PDFs")
    require(len(set(tables)) == 5, "table inputs are not five unique files")

    figures_main = len(re.findall(r"\\begin\{figure\*?\}", main))
    figures_appendix = len(re.findall(r"\\begin\{figure\*?\}", appendix))
    tables_main = len(re.findall(r"\\begin\{table\*?\}", main))
    tables_appendix = len(re.findall(r"\\begin\{table\*?\}", appendix))
    require((figures_main, figures_appendix) == (6, 7), "figure split is not 6+7")
    require((tables_main, tables_appendix) == (2, 3), "table split is not 2+3")
    require(r"\onecolumngrid" not in source, "manual one-column override is forbidden")
    require(r"\begin{algorithm}" not in source, "custom algorithm environment is forbidden")

    proofs = re.findall(r"\\begin\{proof\}(.*?)\\end\{proof\}", source, flags=re.S)
    require(len(proofs) == 5, f"expected five full proofs, found {len(proofs)}")
    require(
        all(word_count(proof) >= 100 for proof in proofs),
        "a proof is too short to be a full derivation",
    )
    require(
        main.count(r"\section*{Limitations}") == 1,
        "exactly one unnumbered Limitations section is required",
    )
    require(
        main.count(r"\section*{Data Availability}") == 1,
        "exactly one unnumbered Data Availability section is required",
    )
    require(
        main.count(r"\section*{Code Availability}") == 1,
        "exactly one unnumbered Code Availability section is required",
    )
    return {
        "main_sections": len(sections),
        "subsections": subsection_counts,
        "figures": len(figures),
        "tables": len(tables),
        "proofs": len(proofs),
    }


def bibliography_keys(bibliography: str) -> tuple[str, ...]:
    return tuple(
        re.findall(r"^@\w+\s*\{\s*([^,\s]+)\s*,", bibliography, flags=re.M)
    )


def citation_report(source: str, bibliography: str) -> dict[str, int]:
    bib_keys = bibliography_keys(bibliography)
    require(len(bib_keys) == len(set(bib_keys)), "duplicate bibliography keys")
    require(len(bib_keys) == 66, f"bibliography has {len(bib_keys)} entries")
    groups = re.findall(
        r"\\cite(?:p|t|alp|alt|author|year|yearpar)?"
        r"(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^{}]+)\}",
        source,
    )
    require(60 <= len(groups) <= 70, f"manuscript has {len(groups)} citation commands")
    group_sizes = tuple(
        len([key for key in group.split(",") if key.strip()]) for group in groups
    )
    require(all(1 <= size <= 3 for size in group_sizes), "citation group size is not 1--3")
    cited = {
        key.strip()
        for group in groups
        for key in group.split(",")
        if key.strip()
    }
    require(len(cited) == 66, f"manuscript cites {len(cited)} unique works")
    require(not (cited - set(bib_keys)), "citation key is missing from bibliography")
    require(not (set(bib_keys) - cited), "bibliography contains an uncited entry")
    require(r"\nocite" not in source, "nocite is not permitted")
    return {
        "bibliography_entries": len(bib_keys),
        "citation_commands": len(groups),
        "citation_mentions": sum(group_sizes),
        "unique_cited_works": len(cited),
    }


def validate_manifest() -> dict[str, int]:
    require(MANIFEST.is_file(), "amortized asset manifest is missing")
    require(sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256, "asset manifest differs")
    manifest = json.loads(MANIFEST.read_text())
    require(manifest.get("schema") == 1, "asset manifest schema is not 1")
    asset_hashes = manifest.get("asset_hashes")
    source_hashes = manifest.get("source_hashes")
    require(isinstance(asset_hashes, dict), "asset_hashes is not an object")
    require(isinstance(source_hashes, dict), "source_hashes is not an object")
    require(len(asset_hashes) == 23, "asset manifest does not contain 23 generated assets")
    require(len(source_hashes) == 4, "asset manifest does not contain four source hashes")
    for relative, expected in {**asset_hashes, **source_hashes}.items():
        path = ROOT / relative
        require(path.is_file(), f"manifest file is missing: {relative}")
        require(
            re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            f"manifest digest is malformed: {relative}",
        )
        require(sha256(path) == expected, f"manifest hash differs: {relative}")
    return {
        "manifest_schema": manifest["schema"],
        "manifest_assets": len(asset_hashes),
        "manifest_sources": len(source_hashes),
    }


def validate_compiled_artifacts(bib_keys: tuple[str, ...]) -> dict[str, object]:
    report: dict[str, object] = {
        "pdf_checked": PDF.is_file(),
        "bbl_checked": BBL.is_file(),
    }
    if PDF.is_file():
        data = PDF.read_bytes()
        require(len(data) > 100_000, "compiled PDF is unexpectedly small")
        require(data.startswith(b"%PDF-"), "compiled PDF header is invalid")
        require(b"%%EOF" in data[-2048:], "compiled PDF terminator is missing")
        report["pdf_bytes"] = len(data)
        report["pdf_sha256"] = hashlib.sha256(data).hexdigest()
    if BBL.is_file():
        bbl = BBL.read_text()
        bbl_keys = tuple(
            re.findall(
                r"\\bibitem\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}",
                bbl,
                flags=re.S,
            )
        )
        require(len(bbl_keys) == 66, f"compiled bibliography has {len(bbl_keys)} items")
        require(len(bbl_keys) == len(set(bbl_keys)), "compiled bibliography repeats a key")
        require(set(bbl_keys) == set(bib_keys), "compiled bibliography keys are stale")
        report["bbl_items"] = len(bbl_keys)
        report["bbl_sha256"] = sha256(BBL)
    return report


def validate_archive(required: bool) -> dict[str, object]:
    if not required:
        return {"archive_checked": False}
    require(ARCHIVE.is_file(), "submission archive is missing")
    sources = archive_sources()
    expected_names = sorted(sources)
    with zipfile.ZipFile(ARCHIVE) as handle:
        require(handle.testzip() is None, "submission archive has a corrupt member")
        infos = handle.infolist()
        names = [info.filename for info in infos]
        require(names == expected_names, "submission archive member order or set differs")
        require(len(names) == len(set(names)) == 22, "archive does not have 22 unique files")
        require("main.bbl" in names, "compiled bibliography is missing from archive")
        for info in infos:
            require(info.date_time == (1980, 1, 1, 0, 0, 0), "archive timestamp differs")
            require(info.create_system == 3, "archive creator platform differs")
            require(info.external_attr >> 16 == 0o100644, "archive file mode differs")
            require(
                info.compress_type == zipfile.ZIP_DEFLATED,
                "archive compression method differs",
            )
        for name, path in sources.items():
            require(path.is_file(), f"archive input is missing: {name}")
            require(handle.read(name) == path.read_bytes(), f"archive member is stale: {name}")
    return {
        "archive_checked": True,
        "archive_members": len(sources),
        "archive_sha256": sha256(ARCHIVE),
    }


def validate(*, require_archive: bool = True) -> dict[str, object]:
    require(MAIN.is_file(), "main.tex is missing")
    require(BIB.is_file(), "refs.bib is missing")
    require(BBL.is_file(), "main.bbl is missing")
    require(POPULAR.is_file(), "PRX popular summary is missing")
    require(sha256(BIB) == EXPECTED_BIB_SHA256, "refs.bib differs from preserved final")

    tex = without_comments(MAIN.read_text())
    bib = BIB.read_text()
    searchable = (tex + "\n" + bib).lower()
    for forbidden in FORBIDDEN_TEXT:
        require(forbidden not in searchable, f"forbidden manuscript text found: {forbidden}")

    documentclass = re.search(
        r"\\documentclass(?:\[([^\]]*)\])?\{([^{}]+)\}", tex
    )
    require(documentclass is not None, "document class is missing")
    options = tuple(
        option.strip()
        for option in (documentclass.group(1) if documentclass else "").split(",")
        if option.strip()
    )
    require(
        documentclass is not None
        and documentclass.group(2) == "revtex4-2"
        and options == ("aps", "prx", "reprint", "superscriptaddress", "floatfix"),
        "exact PRX REVTeX document class is missing",
    )
    require(
        r"\bibliographystyle{apsrev4-2}" in tex,
        "APS bibliography style is missing",
    )
    require(
        r"\renewcommand{\thesection}" not in tex
        and r"\renewcommand{\thesubsection}" not in tex,
        "manual section-number overrides are forbidden",
    )
    popular_words = word_count(POPULAR.read_text())
    require(
        50 <= popular_words <= 150,
        f"PRX popular summary has {popular_words} words",
    )
    require(normalized_title(tex) == EXPECTED_TITLE, "visible title differs")
    require(f"pdftitle={{{EXPECTED_TITLE}}}" in tex, "PDF metadata title differs")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S)
    require(abstract is not None, "abstract is missing")
    abstract_words = word_count(abstract.group(1)) if abstract else 0
    require(abstract_words >= 100, f"abstract has only {abstract_words} words")

    figures, tables = included_paths(tex)
    for relative in (*figures, *tables):
        path = PAPER / relative
        require(path.is_file(), f"included source is missing: {relative}")
    labels = re.findall(r"\\label\{([^{}]+)\}", tex)
    require(len(labels) == len(set(labels)), "duplicate LaTeX labels")
    refs = re.findall(r"\\(?:eq)?ref\{([^{}]+)\}", tex)
    require(not (set(refs) - set(labels)), "reference points to a missing label")

    bib_keys = bibliography_keys(bib)
    return {
        "main_sha256": sha256(MAIN),
        "bibliography_sha256": sha256(BIB),
        "abstract_words": abstract_words,
        "popular_summary_words": popular_words,
        "included_sources": len(figures) + len(tables),
        **structural_report(tex),
        **citation_report(tex, bib),
        **validate_manifest(),
        **validate_compiled_artifacts(bib_keys),
        **validate_archive(require_archive),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help="validate source and generated assets without requiring the zip",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate(require_archive=not args.skip_archive),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
