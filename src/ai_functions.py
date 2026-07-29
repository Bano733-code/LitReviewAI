from app import client
def get_summary(text):
    """Simple summary of abstract (concise)."""
    if not text:
        return "No abstract available."
    prompt = f"""
You are an academic assistant. Summarize the abstract concisely in 3-5 sentences, focusing on contributions, methods and main findings. Use simple language suitable for a researcher.
Abstract:
{text}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def chunk_text(text, max_chars=4000):
    """Split text into smaller parts safely for Groq API."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def get_section_summaries(text):
    summaries = []
    chunks = chunk_text(text)
    for chunk in chunks:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Summarize this section:\n{chunk}"}],
            temperature=0.3,
        )
        summaries.append(response.choices[0].message.content)
    return "\n\n".join(summaries)

def get_limitations(text):
    if not text:
        return "No abstract available."
    prompt = f"Extract only the limitations or challenges discussed in this abstract (if any). If none, write 'No explicit limitations mentioned'.\n\n{text}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def get_research_gaps(text):
    """Return 2-4 potential research gaps based on abstract."""
    if not text:
        return "No abstract available."
    prompt = f"""
You are a research assistant. Based on this abstract, list 2–4 realistic research gaps or unexplored questions that follow logically from the study. Provide concise bullet points.
Abstract:
{text}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


