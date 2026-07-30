import re
import bibtexparser
def export_bibtex(papers):
    db = {"entries": []}
    for i, p in enumerate(papers):
        title = p.get("title") or f"paper_{i}"
        authors = " and ".join(p.get("authors", ["Unknown"]))
        entry = {
            "ENTRYTYPE": "article",
            "ID": re.sub(r"\s+", "_", title)[:80],
            "title": title,
            "author": authors,
            "year": p.get("year", "2025"),
            "abstract": p.get("abstract", "")
        }
        db["entries"].append(entry)
    return bibtexparser.dumps(db)

