import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from mcp.server.fastmcp import FastMCP
from newsplease import NewsPlease

mcp = FastMCP("article")


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def fetch_single_article(url: str) -> Dict[str, str]:
    """
    Fetch and parse a single news article using news-please
    """
    try:
        article = NewsPlease.from_url(url)
        article_dict = article.get_dict()

        # Get article components
        title = article_dict.get("title", "")
        text = article_dict.get("maintext", "")
        authors = article_dict.get("authors", [])
        date = article_dict.get("date_publish", "")
        html = article_dict.get("html", "")

        return {
            "title": title,
            "content": text,
            "authors": ", ".join(authors) if authors else "Unknown",
            "published_date": date.isoformat() if date else "Unknown",
            "origin": url,
            "status": "success",
            "html": html,
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
