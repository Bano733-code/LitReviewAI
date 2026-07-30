from src.config import client


def get_paper_analysis(abstract):
    """
    Generate summary, limitations, and research gaps
    in a single API call.
    """

    if not abstract:
        return {
            "summary": "No abstract available.",
            "limitations": "N/A",
            "research_gaps": "N/A"
        }

    prompt = f"""
You are an academic research assistant.

Analyze the following research abstract and return your answer in exactly this format:

SUMMARY:
(3-5 sentence summary)

LIMITATIONS:
- bullet point
- bullet point

RESEARCH_GAPS:
- bullet point
- bullet point

Abstract:
{abstract}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
    )

    text = response.choices[0].message.content

    summary = ""
    limitations = ""
    research_gaps = ""

    current = None

    for line in text.splitlines():

        line = line.strip()

        if line.upper().startswith("SUMMARY"):
            current = "summary"
            continue

        elif line.upper().startswith("LIMITATIONS"):
            current = "limitations"
            continue

        elif line.upper().startswith("RESEARCH_GAPS"):
            current = "research_gaps"
            continue

        if current == "summary":
            summary += line + "\n"

        elif current == "limitations":
            limitations += line + "\n"

        elif current == "research_gaps":
            research_gaps += line + "\n"

    return {
        "summary": summary.strip(),
        "limitations": limitations.strip(),
        "research_gaps": research_gaps.strip()
    }