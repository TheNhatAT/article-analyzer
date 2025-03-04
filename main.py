import json
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from newsplease import NewsPlease
import requests

mcp = FastMCP("article")


def clean_html(html: str) -> str:
    """
    Clean HTML content by removing unnecessary elements and attributes.

    Args:
        html (str): Raw HTML content
    Returns:
        str: Cleaned HTML content
    """
    if not html:
        return ""

    try:
        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Remove unnecessary tags
        for tag in soup.find_all(
            ["script", "style", "iframe", "nav", "footer", "header", "meta", "link"]
        ):
            tag.decompose()

        # Remove common ad-related elements
        for element in soup.find_all(
            class_=re.compile(
                r"(ad|advertisement|banner|popup|modal|cookie|subscribe|newsletter)"
            )
        ):
            element.decompose()

        # Remove empty elements
        for element in soup.find_all():
            if len(element.get_text(strip=True)) == 0:
                element.decompose()

        # Remove all style attributes
        for tag in soup.find_all(True):
            tag.attrs = {
                key: value
                for key, value in tag.attrs.items()
                if key not in ["style", "class", "id"]
            }

        return str(soup)
    except Exception as e:
        print(f"Error cleaning HTML: {str(e)}")
        return html


def fetch_html(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch HTML content from URL separately.

    Args:
        url (str): URL to fetch
    Returns:
        Tuple[Optional[str], Optional[str]]: Tuple of (raw HTML, error message if any)
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text, None
    except requests.RequestException as e:
        return None, f"Failed to fetch HTML: {str(e)}"


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def fetch_single_article(url: str) -> Dict[str, str]:
    """
    Fetch and parse a single news article using news-please

    Args:
        url (str): URL of the article to fetch
    Returns:
        Dict[str, str]: Article information including parsed content and raw HTML
    """
    try:
        # Fetch raw HTML first
        raw_html, error = fetch_html(url)
        if error:
            raise Exception(error)

        # Parse article using NewsPlease
        article = NewsPlease.from_url(url)
        if not article:
            raise Exception("Failed to parse article")

        article_dict = article.get_dict()

        # Clean and process HTML
        cleaned_html = clean_html(raw_html)

        # Get article components
        title = article_dict.get("title", "").strip()
        text = article_dict.get("maintext", "").strip()
        authors = article_dict.get("authors", [])
        date = article_dict.get("date_publish", "")

        return {
            "title": title or "Untitled",
            "content": text,
            "authors": ", ".join(authors) if authors else "Unknown",
            "published_date": date.isoformat() if date else "Unknown",
            "origin": url,
            "status": "success",
            "html": cleaned_html,
        }
    except Exception as e:
        return {
            "title": "Error",
            "content": f"Error fetching article: {str(e)}",
            "authors": "N/A",
            "published_date": "N/A",
            "origin": url,
            "status": "error",
            "html": "",
        }


@mcp.tool()
def fetch_article(url: str) -> Dict[str, str]:
    """
    Fetch and parse a news article from a given URL

    Args:
        url (str): The URL of the article to fetch
    Returns:
        Dict[str, str]: Article information including title, content, origin
    """
    return fetch_single_article(url)


@mcp.tool()
def fetch_articles(urls: List[str]) -> List[Dict[str, str]]:
    """
    Fetch and parse multiple news articles in parallel using ThreadPoolExecutor

    Args:
        urls (List[str]): List of URLs to fetch
    Returns:
        List[Dict[str, str]]: List of article information dictionaries
    """
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Map URLs to fetch_single_article function
        results = list(executor.map(fetch_single_article, urls))
    return results


if __name__ == "__main__":
    mcp.run(transport="stdio")
