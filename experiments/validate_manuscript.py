#!/usr/bin/env python3
"""Validate the evidence-faithful RSQ manuscript and its submission archive."""

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
PDF = PAPER / "main.pdf"
ARCHIVE = PAPER / "arxiv-source-rsq.zip"
MANIFEST = PAPER / "tables" / "amortized_asset_manifest.json"
EXPECTED_TITLE = (
    "Task-Weighted QAOA Subspaces: Local Optimality without a Matched-SPSA "
    "Query Saving"
)
EXPECTED_AUTHOR = "Molena Huynh"
EXPECTED_AFFILIATION = "North Carolina State University"
EXPECTED_EMAIL = "molena.huynh@jmp.com"
EXPECTED_SECTIONS = ["Results", "Discussion", "Methods"]
EXPECTED_RESULTS_SUBSECTIONS = [
    "Multi-Angle QAOA",
    "Repeated Weighted Objectives",
    "Randomized Range Approximation",
    "Relation to Prior Work",
    "Limits of Static and Unweighted Subspaces",
    "Task-Weighted Observable Subspaces",
    "Development Experiments",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def main_text_word_count(source: str) -> int:
    """Approximate the NMI main-text count, excluding displays and end matter."""
    abstract_end = source.find(r"\end{abstract}")
    start = abstract_end + len(r"\end{abstract}")
    end = source.find(r"\section{Methods}")
    require(abstract_end >= 0 and end > start, "main-text boundaries are missing")
    main = source[start:end]
    main = re.sub(
        r"\\begin\{(?:figure|table)\*?\}.*?\\end\{(?:figure|table)\*?\}",
        " ",
        main,
        flags=re.S,
    )
    main = re.sub(
        r"\\begin\{(?:equation|align|aligned)\*?\}.*?"
        r"\\end\{(?:equation|align|aligned)\*?\}",
        " ",
        main,
        flags=re.S,
    )
    return word_count(main)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def included_sources(source: str) -> dict[str, Path]:
    source = without_comments(source)
    relative = set(
        re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", source)
    )
    relative.update(re.findall(r"\\input\{([^{}]+)\}", source))
    return {name: PAPER / name for name in sorted(relative)}


def structural_report(source: str) -> dict[str, object]:
    split = source.find(r"\onecolumn")
    require(split >= 0, "one-column appendix transition is missing")
    main = source[:split]
    appendix = source[split:]
    section_matches = list(re.finditer(r"^\\section\{([^{}]+)\}", main, flags=re.M))
    sections = [match.group(1) for match in section_matches]
    require(sections == EXPECTED_SECTIONS, f"main sections differ: {sections}")
    require(
        r"\section{Introduction}" not in main,
        "Introduction must be unheaded for the NMI Article structure",
    )

    section_bodies: dict[str, str] = {}
    for index, match in enumerate(section_matches):
        end = section_matches[index + 1].start() if index + 1 < len(
            section_matches
        ) else len(main)
        section_bodies[match.group(1)] = main[match.end():end]
    results_subsections = re.findall(
        r"^\\subsection\{([^{}]+)\}",
        section_bodies["Results"],
        flags=re.M,
    )
    require(
        results_subsections == EXPECTED_RESULTS_SUBSECTIONS,
        f"Results subsections differ: {results_subsections}",
    )
    require(
        not re.findall(
            r"^\\subsection\{", section_bodies["Discussion"], flags=re.M
        ),
        "Discussion must not contain a stray subsection",
    )
    require(
        re.findall(
            r"^\\subsection\{([^{}]+)\}",
            section_bodies["Methods"],
            flags=re.M,
        ) == ["Use of generative AI tools"],
        "Methods subsection structure differs",
    )
    require(
        len(re.findall(r"^\\subsubsection\{", section_bodies["Results"], flags=re.M))
        == 10,
        "Results must contain the ten declared third-level headings",
    )

    figures_main = len(re.findall(r"\\begin\{figure\*?\}", main))
    figures_appendix = len(re.findall(r"\\begin\{figure\*?\}", appendix))
    tables_main = len(re.findall(r"\\begin\{table\*?\}", main))
    tables_appendix = len(re.findall(r"\\begin\{table\*?\}", appendix))
    main_displays = figures_main + tables_main
    appendix_displays = figures_appendix + tables_appendix
    require(main_displays <= 6, f"main text has {main_displays} display items")
    require(
        appendix_displays <= 10,
        f"appendix has {appendix_displays} display items",
    )
    for command in (
        r"\setcounter{figure}{0}",
        r"\setcounter{table}{0}",
        r"\renewcommand{\figurename}{Extended Data Figure}",
        r"\renewcommand{\tablename}{Extended Data Table}",
        r"\renewcommand{\theHfigure}{extended-data.figure.\arabic{figure}}",
        r"\renewcommand{\theHtable}{extended-data.table.\arabic{table}}",
    ):
        require(command in appendix, f"Extended Data invariant missing: {command}")
    require(
        r"Extended Data Figures~\ref{fig:operatorbudget}--\ref{fig:reproductionmap}"
        in main,
        "main-text Extended Data figure cross-reference is missing",
    )
    require(
        r"Extended Data Tables~\ref{tab:development-gate}--\ref{tab:shot}"
        in main,
        "main-text Extended Data table cross-reference is missing",
    )
    require(source.count(r"\begin{algorithm}") == 1, "expected one algorithm")
    require(source.count(r"\begin{proof}") >= 4, "expected at least four full proofs")
    require(r"\paragraph{Limitations.}" in main, "limitations paragraph is missing")
    require(
        not re.search(r"\\section\*?\{Acknowledg(?:e)?ments?\}", source, flags=re.I),
        "acknowledgements must be omitted",
    )
    require(
        not re.search(r"\\section\*?\{Funding\}", source, flags=re.I),
        "funding section must be omitted",
    )
    require("impact statement" not in source.lower(), "impact statement must be omitted")
    return {
        "main_sections": len(sections),
        "results_subsections": len(results_subsections),
        "results_subsubsections": 10,
        "main_display_items": main_displays,
        "appendix_display_items": appendix_displays,
        "extended_data_items": appendix_displays,
        "figures": figures_main + figures_appendix,
        "tables": tables_main + tables_appendix,
        "algorithms": 1,
        "proofs": source.count(r"\begin{proof}"),
    }


def citation_report(source: str, bibliography: str) -> dict[str, int]:
    bib_keys = re.findall(r"^@\w+\s*\{\s*([^,\s]+)\s*,", bibliography, flags=re.M)
    require(len(bib_keys) == len(set(bib_keys)), "duplicate bibliography keys")
    groups = re.findall(
        r"\\cite(?:p|t|alp|alt|author|year|yearpar)?"
        r"(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^{}]+)\}",
        source,
    )
    group_sizes = [len([key for key in group.split(",") if key.strip()]) for group in groups]
    require(all(1 <= size <= 3 for size in group_sizes), "citation group exceeds three works")
    cited = {
        key.strip()
        for group in groups
        for key in group.split(",")
        if key.strip()
    }
    require(not (cited - set(bib_keys)), "citation key is missing from bibliography")
    require(len(cited) <= 50, f"visible reference list has {len(cited)} works")
    require("huynh2026gctr" in cited, "GCTR arXiv v1 citation is missing")
    require("2604.24803" in bibliography, "GCTR arXiv identifier is missing")
    require("2604.24803v1 [cs.LG]" in bibliography, "GCTR version/class is missing")
    forbidden_identifier = "2607." + "06758"
    require(
        forbidden_identifier not in source + bibliography,
        "forbidden arXiv citation present",
    )
    require(r"\nocite" not in source, "nocite is not permitted")
    return {
        "bibliography_database_entries": len(bib_keys),
        "citation_commands": len(groups),
        "citation_mentions": sum(group_sizes),
        "unique_cited_works": len(cited),
    }


def validate_manifest() -> dict[str, int]:
    require(MANIFEST.is_file(), "display manifest is missing")
    manifest = json.loads(MANIFEST.read_text())
    asset_hashes = manifest.get("asset_hashes", {})
    source_hashes = manifest.get("source_hashes", {})
    require(asset_hashes, "display manifest contains no asset hashes")
    for relative, expected in {**asset_hashes, **source_hashes}.items():
        path = ROOT / relative
        require(path.is_file(), f"manifest file is missing: {relative}")
        require(sha256(path) == expected, f"manifest hash differs: {relative}")
    return {
        "manifest_assets": len(asset_hashes),
        "manifest_sources": len(source_hashes),
    }


def archive_sources(source: str | None = None) -> dict[str, Path]:
    if source is None:
        source = without_comments(MAIN.read_text())
    sources = {"main.tex": MAIN, "main.bbl": BBL, "refs.bib": BIB}
    sources.update(included_sources(source))
    return sources


def validate_archive(source: str, required: bool) -> dict[str, object]:
    if not required:
        return {"archive_checked": False}
    require(ARCHIVE.is_file(), "submission archive is missing")
    sources = archive_sources(source)
    with zipfile.ZipFile(ARCHIVE) as handle:
        names = [name for name in handle.namelist() if not name.endswith("/")]
        require(set(names) == set(sources), "submission archive member set differs")
        for name, path in sources.items():
            require(path.is_file(), f"archive input is missing: {name}")
            require(handle.read(name) == path.read_bytes(), f"archive member is stale: {name}")
    return {
        "archive_checked": True,
        "archive_members": len(sources),
        "archive_sha256": sha256(ARCHIVE),
    }


def validate(*, require_archive: bool = True) -> dict[str, object]:
    tex = without_comments(MAIN.read_text())
    bib = BIB.read_text()
    require(r"\documentclass[10pt,twocolumn]{article}" in tex, "two-column format is missing")
    require(normalized_title(tex) == EXPECTED_TITLE, "visible title differs")
    require(f"pdftitle={{{EXPECTED_TITLE}}}" in tex, "PDF metadata title differs")
    require(EXPECTED_AUTHOR in tex, "author name differs")
    require(EXPECTED_AFFILIATION in tex, "author affiliation differs")
    require(EXPECTED_EMAIL in tex, "author email differs")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S)
    require(abstract is not None, "abstract is missing")
    abstract_words = word_count(abstract.group(1)) if abstract else 0
    require(100 <= abstract_words <= 150, f"abstract has {abstract_words} words")
    main_words = main_text_word_count(tex)
    require(main_words <= 3500, f"main text has approximately {main_words} words")
    for heading in (
        "Data availability",
        "Code availability",
        "Author contributions",
        "Competing interests",
    ):
        require(
            rf"\section*{{{heading}}}" in tex,
            f"{heading} section is missing",
        )
    require(r"\section{Methods}" in tex, "Methods section is missing")
    require(
        r"\subsection{Use of generative AI tools}" in tex,
        "generative-AI Methods subsection is missing",
    )
    require("OpenAI Codex" in tex, "OpenAI Codex disclosure is missing")
    require(
        "No large language model is listed as an author." in tex,
        "LLM authorship statement is missing",
    )
    require(
        re.search(r"unmatched\s+random-basis control", tex) is not None,
        "random-basis caveat is missing",
    )
    require(
        re.search(r"one\s+observed obstacle", tex) is not None,
        "refresh causal caveat is missing",
    )
    require(
        all(term in tex for term in ("incomplete", "unregistered", "unpowered", "unexecuted")),
        "prospective-design status is incomplete",
    )
    require(
        "not an end-to-end hardware experiment" in tex,
        "finite-shot hardware caveat is missing",
    )
    require(
        re.search(r"one evaluation\s+realization per graph-depth cell", tex)
        is not None,
        "exact-track repeat description is missing",
    )
    require(
        re.search(r"81\s+optimization-objective calls per task", tex) is not None,
        "matched SPSA query ledger is missing",
    )
    availability = re.search(
        r"\\section\*\{Data availability\}(.*?)"
        r"\\section\*\{Author contributions\}",
        tex,
        flags=re.S,
    )
    require(availability is not None, "availability statements are malformed")
    availability_text = availability.group(1) if availability else ""
    availability_normalized = " ".join(availability_text.lower().split())
    require(
        len(re.findall(
            r"public versioned(?: repository)? release and archival doi "
            r"are pending",
            availability_normalized,
        )) == 2,
        "data/code availability must state release-pending status twice",
    )
    require(
        "github.com" not in availability_text.lower(),
        "availability statements claim an existing public release",
    )

    included = included_sources(tex)
    for name, path in included.items():
        require(path.is_file(), f"included source is missing: {name}")
    labels = re.findall(r"\\label\{([^{}]+)\}", tex)
    require(len(labels) == len(set(labels)), "duplicate LaTeX labels")
    refs = re.findall(r"\\(?:eq)?ref\{([^{}]+)\}", tex)
    require(not (set(refs) - set(labels)), "reference points to a missing label")
    require(PDF.is_file() and PDF.stat().st_size > 100_000, "compiled PDF is missing")
    require(PDF.stat().st_mtime >= MAIN.stat().st_mtime, "compiled PDF is older than source")

    report = {
        "abstract_words": abstract_words,
        "approximate_main_text_words": main_words,
        "included_sources": len(included),
        **structural_report(tex),
        **citation_report(tex, bib),
        **validate_manifest(),
        **validate_archive(tex, require_archive),
    }
    if BBL.is_file():
        bibitems = len(re.findall(r"^\\bibitem", BBL.read_text(), flags=re.M))
        require(
            bibitems == report["unique_cited_works"],
            "compiled bibliography is stale",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-archive", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        validate(require_archive=not args.skip_archive),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
