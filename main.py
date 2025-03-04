import json
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Set
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from newsplease import NewsPlease
from newsplease.crawler.simple_crawler import SimpleCrawler
import requests
import urllib.parse

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


def extract_links(html: str, base_url: str) -> Set[str]:
    """
    Extract links from HTML content that belong to the same domain.

    Args:
        html (str): HTML content
        base_url (str): Base URL to filter links
    Returns:
        Set[str]: Set of extracted links
    """
    if not html:
        return set()

    # Parse the base URL to get domain
    parsed_base = urllib.parse.urlparse(base_url)
    base_domain = parsed_base.netloc

    try:
        soup = BeautifulSoup(html, "html.parser")
        links = set()

        # Find all <a> tags with href attribute
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Handle relative URLs
            if href.startswith("/"):
                full_url = f"{parsed_base.scheme}://{base_domain}{href}"
                links.add(full_url)
            # Handle absolute URLs from same domain
            elif href.startswith(("http://", "https://")):
                parsed_href = urllib.parse.urlparse(href)
                # Only include links from same domain
                if parsed_href.netloc == base_domain:
                    links.add(href)

        return links
    except Exception as e:
        print(f"Error extracting links: {str(e)}")
        return set()


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


@mcp.tool()
def fetch_recursive(
    url: str, depth: int = 1, max_articles: int = 5
) -> List[Dict[str, str]]:
    """
    Recursively fetch articles from a website up to specified depth.

    Args:
        url (str): The starting URL to fetch
        depth (int): How many levels deep to crawl (default: 1)
        max_articles (int): Maximum number of articles to fetch (default: 5)
    Returns:
        List[Dict[str, str]]: List of article information dictionaries
    """
    if depth < 1 or max_articles < 1:
        return [{"error": "Depth and max_articles must be greater than 0"}]

    # First, fetch the main article
    main_article = fetch_single_article(url)
    results = [main_article]

    # If we don't need to go deeper, return early
    if depth == 1 or len(results) >= max_articles:
        return results[:max_articles]

    # Extract links from the main article HTML
    raw_html, _ = fetch_html(url)
    if not raw_html:
        return results[:max_articles]

    cleaned_html = clean_html(raw_html)

    links = extract_links(cleaned_html, url)
    links_to_crawl = list(links)[
        :max_articles
    ]  # Limit links to prevent excessive crawling

    # Fetch articles from extracted links
    with ThreadPoolExecutor(max_workers=min(10, len(links_to_crawl))) as executor:
        link_results = list(executor.map(fetch_single_article, links_to_crawl))

    # Add non-error results
    for article in link_results:
        if article["status"] == "success" and len(results) < max_articles:
            results.append(article)

    return results[:max_articles]


@mcp.tool()
def fetch_sitemap_articles(url: str, max_articles: int = 10) -> List[Dict[str, str]]:
    """
    Fetch articles from a website's sitemap.

    Args:
        url (str): The website URL to fetch articles from
        max_articles (int): Maximum number of articles to fetch (default: 10)
    Returns:
        List[Dict[str, str]]: List of article information dictionaries
    """
    try:
        # Use SimpleCrawler for sitemap-based crawling
        articles = SimpleCrawler.fetch_urls(
            urls=[url],
            overwrite_existing_files=True,
            sitemap_crawler=True,
            number_of_articles=max_articles,
        )

        results = []
        for article_url, article in articles.items():
            if article:
                article_dict = article.get_dict()
                results.append(
                    {
                        "title": article_dict.get("title", "").strip() or "Untitled",
                        "content": article_dict.get("maintext", "").strip(),
                        "authors": ", ".join(article_dict.get("authors", []))
                        or "Unknown",
                        "published_date": article_dict.get("date_publish", "Unknown"),
                        "origin": article_url,
                        "status": "success",
                    }
                )

                # Limit to max_articles
                if len(results) >= max_articles:
                    break

        return results
    except Exception as e:
        return [{"error": f"Error fetching sitemap articles: {str(e)}"}]


if __name__ == "__main__":
    mcp.run(transport="stdio")
