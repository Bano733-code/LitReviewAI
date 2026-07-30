import requests
import xml.etree.ElementTree as ET

GROBID_URL = (
    "https://grobid.science-miner.com/api/processFulltextDocument"
)


def extract_metadata_grobid(pdf_file):
    """
    Extract title, authors and abstract using GROBID.
    Returns a dictionary.
    """

    pdf_file.seek(0)

    response = requests.post(
        GROBID_URL,
        files={
            "input": pdf_file
        },
        headers={
            "Accept": "application/xml"
        },
        timeout=60
    )

    if response.status_code != 200:
        raise Exception("GROBID request failed")

    root = ET.fromstring(response.text)

    ns = {
        "tei": "http://www.tei-c.org/ns/1.0"
    }

    # ------------------------
    # Title
    # ------------------------

    title = ""

    title_node = root.find(
        ".//tei:titleStmt/tei:title",
        ns
    )

    if title_node is not None and title_node.text:
        title = title_node.text.strip()

    # ------------------------
    # Authors
    # ------------------------

    authors = []

    for author in root.findall(".//tei:author", ns):

        first = author.find(
            ".//tei:forename",
            ns
        )

        last = author.find(
            ".//tei:surname",
            ns
        )

        first_name = first.text.strip() if (
            first is not None and first.text
        ) else ""

        last_name = last.text.strip() if (
            last is not None and last.text
        ) else ""

        full_name = (
            first_name + " " + last_name
        ).strip()

        if full_name:
            authors.append(full_name)

    # ------------------------
    # Abstract
    # ------------------------

    abstract = ""

    abs_node = root.find(
        ".//tei:abstract",
        ns
    )

    if abs_node is not None:

        abstract = " ".join(
            abs_node.itertext()
        ).strip()

    return {

        "title": title,

        "authors": authors,

        "abstract": abstract

    }