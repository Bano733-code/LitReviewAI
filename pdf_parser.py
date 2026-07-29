def extract_text_from_pdf(pdf_file):
    """Extract all text from PDF. Resets pointer outside if needed."""
    data = pdf_file.read()
    pdf_file.seek(0)
    with fitz.open(stream=data, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text("text")
    return text