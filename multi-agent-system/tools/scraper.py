"""
Web page scraper tool — fetches a URL and returns clean readable text.
Strips scripts, styles, nav, and other noise.
"""
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
MAX_CHARS = 6000  # cap per page to stay within context limits


def read_page(url: str) -> str:
    """
    Fetch a web page and return its main text content (up to MAX_CHARS characters).
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Error fetching {url}: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    # Prefer <article> or <main> if present, otherwise use <body>
    content = soup.find("article") or soup.find("main") or soup.find("body")
    if content is None:
        return "Could not extract content from page."

    text = content.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    lines = [line for line in text.splitlines() if line.strip()]
    clean = "\n".join(lines)

    return clean[:MAX_CHARS] + ("..." if len(clean) > MAX_CHARS else "")


SCRAPER_TOOL_SCHEMA = {
    "name": "read_page",
    "description": (
        "Fetch and read the full text content of a web page at a given URL. "
        "Use this after web_search to read the details of promising results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL of the page to read.",
            }
        },
        "required": ["url"],
    },
}
