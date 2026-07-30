import re 
from src.pdf_parser import extract_text_from_pdf
def extract_title(text):
    """Extracts paper title (before Abstract), avoiding citations and irrelevant headers."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Find index of "Abstract" (case-insensitive) or common translations
    abs_index = None
    for i, line in enumerate(lines):
        low = line.lower()
        if low.startswith("abstract") or any(k in low for k in ["résumé", "resumen", "summary", "overview", "abstract—"]):
            abs_index = i
            break

    # If Abstract found, check lines before abstract; else check top area
    if abs_index is not None:
        candidate_lines = lines[:abs_index]
    else:
        candidate_lines = lines[:100]

    skip_words = ["journal", "doi", "copyright", "arxiv", "volume",
                  "methods", "open access", "citation", "editor", "published"]

    candidates = []
    for line in candidate_lines:
        low = line.lower()
        # Skip unwanted lines
        if any(w in low for w in skip_words):
            continue
        # Skip author-like lists (many commas) or emails
        if "@" in line or re.search(r"\bjournal\b", low):
            continue
        # Skip short/very long lines
        if 5 <= len(line.split()) <= 30:
            # avoid lines that look like "Editor: Name" etc
            if re.match(r"^(editor|edited by|edited).+", low):
                continue
            candidates.append(line)

    # Prefer line with colon (typical title: subtitle)
    for line in candidates:
        if ":" in line:
            return line

    # fallback: longest candidate
    if candidates:
        return max(candidates, key=len)
    return "Untitled"


def extract_authors(text, max_scan_lines=80):
    """
    Flexible author extraction:
    - Finds a likely title, then scans the following lines looking for names.
    - Handles comma-separated authors on one line or names on separate lines.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    authors = []

    # 1) find a likely title index (first reasonably long line)
    title_idx = None
    for i, line in enumerate(lines[:40]):
        if len(line.split()) >= 4 and not any(k in line.lower() for k in ["journal", "doi", "copyright", "arxiv"]):
            title_idx = i
            break
    start = title_idx + 1 if title_idx is not None else 0

    # 2) scan lines after title until we hit abstract/keywords or a limit
    abstract_markers = ["abstract", "résumé", "resumen", "summary", "overview", "introduction", "keywords"]
    for line in lines[start:start + max_scan_lines]:
        low = line.lower()
        if any(m in low for m in abstract_markers):
            break

        # If line contains commas or " and ", likely multiple authors
        if "," in line or " and " in low or ";" in line:
            parts = re.split(r",|;|\sand\s", line)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # match typical name patterns: "First Last", "First M. Last", "Last, First"
                # handle "Last, First" by swapping
                if re.match(r"^[A-Z][a-z]+,\s*[A-Z][a-z]+", part):
                    # swap "Last, First" -> "First Last"
                    t = re.split(r",\s*", part)
                    name = t[1] + " " + t[0]
                else:
                    name = part

                # require at least two capitalized tokens to be considered a name
                cap_tokens = [t for t in name.split() if re.match(r"^[A-Z][a-z]+\.?$", t) or re.match(r"^[A-Z]\.$", t)]
                if len(cap_tokens) >= 2 and not looks_like_affiliation(name):
                    authors.append(name)
            if authors:
                continue
        else:
            # attempt single-line name extraction
            matches = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z]\.?\s?[A-Z][a-z]+){0,2}\b', line)
            for m in matches:
                if len(m.split()) >= 2 and not looks_like_affiliation(m):
                    authors.append(m)

    # 3) fallback: check metadata-like "By X Y" lines
    if not authors:
        for line in lines[:40]:
            m = re.match(r"^(By|BY|by)\s+(.+)$", line)
            if m:
                parts = re.split(r",|;|\sand\s", m.group(2))
                for p in parts:
                    p = p.strip()
                    if p and not looks_like_affiliation(p):
                        authors.append(p)
                if authors:
                    break

    return authors if authors else ["Unknown"]


AFFIL_KEYWORDS = [
    "university", "institute", "department", "school", "hospital",
    "clinic", "center", "centre", "laboratory", "lab", "college",
    "medicine", "research", "faculty", "division", "program", "department of"
]

def looks_like_affiliation(line):
    low = line.lower()
    if any(k in low for k in AFFIL_KEYWORDS):
        return True
    if "@" in line or "http" in low or "www." in low:
        return True
    # if line has few capitalized tokens relative to total tokens, it's likely not a name
    toks = [t for t in line.split() if t.strip()]
    if len(toks) == 0:
        return True
    cap = sum(1 for t in toks if re.match(r"^[A-Z][a-z]+$", t))
    if cap / len(toks) < 0.4:
        return True
    return False


def extract_abstract(text):
    """Extract abstract block (multiple formats supported). Gathers until next main heading."""
    lines = text.split("\n")
    abstract_lines = []
    capture = False
    abstract_markers = ["abstract", "résumé", "resumen", "summary", "overview"]
    stop_markers = ["introduction", "keywords", "1.", "methods", "materials", "results", "conclusion", "references"]

    for line in lines:
        l = line.lower().strip()
        if any(m in l for m in abstract_markers):
            capture = True
            continue
        if capture:
            # stop when a common section heading appears
            if any(re.match(rf"^{sm}\b", l) for sm in stop_markers):
                break
            abstract_lines.append(line.strip())
    abstract = " ".join(abstract_lines).strip()
    return abstract


def extract_metadata(pdf_file):
    """Extract title, authors, abstract from PDF file-like object."""
    text = extract_text_from_pdf(pdf_file)
    title = extract_title(text)
    authors = extract_authors(text)
    abstract = extract_abstract(text)
    return {"title": title or "Untitled", "authors": authors, "abstract": abstract or ""}
