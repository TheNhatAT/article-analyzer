import json
import re
import time
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import List, Dict, Optional, Tuple, Set, Pattern
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from newsplease import NewsPlease
from newsplease.crawler.simple_crawler import SimpleCrawler
import requests
import urllib.parse


# Configuration classes
@dataclass
class TimeoutConfig:
    """Timeout configuration settings"""

    CONNECT_TIMEOUT: int = 5  # seconds for initial connection
    READ_TIMEOUT: int = 10  # seconds for reading response
    NEWSPLEASE_TIMEOUT: int = 10  # seconds for NewsPlease parsing
    TOTAL_REQUEST_TIMEOUT: int = 15  # maximum time for entire request


@dataclass
class RetryConfig:
    """Retry configuration settings"""

    MAX_RETRIES: int = 2
    BACKOFF_FACTOR: float = 0.3
    MAX_WORKERS: int = 10  # for ThreadPoolExecutor


@dataclass
class HTTPConfig:
    """HTTP request configuration"""

    USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    CHUNK_SIZE: int = 8192
    ALLOW_REDIRECTS: bool = True


@dataclass
class ParsingConfig:
    """HTML parsing configuration"""

    UNNECESSARY_TAGS: List[str] = field(
        default_factory=lambda: [
            "script",
            "style",
            "iframe",
            "nav",
            "footer",
            "header",
            "meta",
            "link",
        ]
    )
    UNWANTED_ATTRIBUTES: List[str] = field(
        default_factory=lambda: ["style", "class", "id"]
    )
    AD_PATTERN: str = (
        r"(ad|advertisement|banner|popup|modal|cookie|subscribe|newsletter)"
    )
    ARTICLE_PATTERN: str = r"article|post|content|story"


# Global Configuration
TIMEOUT = TimeoutConfig()
RETRY = RetryConfig()
HTTP = HTTPConfig()
PARSING = ParsingConfig()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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
        for tag in soup.find_all(PARSING.UNNECESSARY_TAGS):
            tag.decompose()

        # Remove common ad-related elements
        for element in soup.find_all(class_=re.compile(PARSING.AD_PATTERN)):
            element.decompose()

        # Remove empty elements
        for element in soup.find_all():
            if len(element.get_text(strip=True)) == 0:
                element.decompose()

        # Remove unwanted attributes
        for tag in soup.find_all(True):
            tag.attrs = {
                key: value
                for key, value in tag.attrs.items()
                if key not in PARSING.UNWANTED_ATTRIBUTES
            }

        return str(soup)
    except Exception as e:
        logger.error(f"Error cleaning HTML: {str(e)}")
        return html


def extract_article_content(html: str) -> Dict[str, str]:
    """
    Extract article content from HTML using BeautifulSoup as fallback.

    Args:
        html (str): Raw HTML content
    Returns:
        Dict[str, str]: Extracted article information
    """
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Try to find article title
        title = None
        title_candidates = [
            soup.find("meta", property="og:title"),
            soup.find("meta", property="twitter:title"),
            soup.find("h1"),
            soup.find("title"),
        ]
        for candidate in title_candidates:
            if candidate:
                title = candidate.get("content", candidate.text)
                if title:
                    break

        # Try to find article content
        content = ""
        main_content = soup.find("article") or soup.find(
            class_=re.compile(PARSING.ARTICLE_PATTERN)
        )
        if main_content:
            # Remove unwanted elements
            for tag in main_content.find_all(PARSING.UNNECESSARY_TAGS):
                tag.decompose()
            content = main_content.get_text(separator="\n", strip=True)
        else:
            # Fallback to largest text block
            paragraphs = soup.find_all("p")
            if paragraphs:
                content = "\n".join(p.get_text(strip=True) for p in paragraphs)

        return {
            "title": title or "Untitled",
            "content": content,
            "status": "success" if content else "error",
        }
    except Exception as e:
        return {"title": "Error", "content": str(e), "status": "error"}


def timeout_decorator(timeout):
    """Decorator to add timeout to a function"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                logger.error(
                    f"Function {func.__name__} timed out after {timeout} seconds"
                )
                thread.join(0)  # Don't wait for thread
                raise TimeoutError(f"Function timed out after {timeout} seconds")

            if error[0] is not None:
                raise error[0]

            return result[0]

        return wrapper

    return decorator


def log_elapsed_time(func):
    """Decorator to log function execution time"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Completed {func.__name__} in {end_time - start_time:.2f} seconds")
        return result

    return wrapper


@log_elapsed_time
def fetch_html(
    url: str,
    total_timeout: int = TIMEOUT.TOTAL_REQUEST_TIMEOUT,
    connect_timeout: int = TIMEOUT.CONNECT_TIMEOUT,
    retries: int = RETRY.MAX_RETRIES,
    backoff_factor: float = RETRY.BACKOFF_FACTOR,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch HTML content from URL with strict timeout control.

    Args:
        url (str): URL to fetch
        total_timeout (int): Maximum total time allowed for request in seconds
        connect_timeout (int): Timeout for initial connection in seconds
        retries (int): Number of retry attempts
        backoff_factor (float): Backoff factor between retries
    Returns:
        Tuple[Optional[str], Optional[str]]: Tuple of (raw HTML, error message if any)
    """
    headers = {"User-Agent": HTTP.USER_AGENT}

    # Create session for connection pooling
    with requests.Session() as session:
        session.headers.update(headers)

        for attempt in range(retries + 1):
            try:
                logger.info(
                    f"Attempting to fetch URL: {url} (Attempt {attempt + 1}/{retries + 1})"
                )
                # Use streaming to monitor download time
                with session.get(
                    url,
                    timeout=(connect_timeout, total_timeout),
                    stream=True,
                    allow_redirects=HTTP.ALLOW_REDIRECTS,
                ) as response:
                    response.raise_for_status()
                    content = []

                    total_chunks = 0
                    total_size = 0
                    start_time = time.time()

                    # Read the content in chunks with total timeout enforcement
                    for chunk in response.iter_content(
                        chunk_size=HTTP.CHUNK_SIZE, decode_unicode=True
                    ):
                        current_time = time.time()
                        elapsed = current_time - start_time
                        if elapsed > total_timeout:
                            logger.warning(
                                f"Request exceeded timeout limit of {total_timeout}s (elapsed: {elapsed:.2f}s)"
                            )
                            return (
                                None,
                                f"Total request time exceeded {total_timeout} seconds",
                            )
                        if chunk:
                            content.append(chunk)
                            total_chunks += 1
                            total_size += len(chunk)
                            if total_chunks % 10 == 0:  # Log every 10 chunks
                                logger.debug(
                                    f"Downloaded {total_size/1024:.1f}KB in {total_chunks} chunks. Elapsed: {elapsed:.2f}s"
                                )

                    return "".join(content), None

            except requests.ConnectTimeout:
                if attempt == retries:
                    logger.error("Connection timed out after all retries")
                    return None, "Connection timed out"
                logger.warning(
                    f"Connection timeout on attempt {attempt + 1}, retrying..."
                )

            except requests.ReadTimeout:
                if attempt == retries:
                    logger.error("Read timed out after all retries")
                    return None, "Read timed out"
                logger.warning(f"Read timeout on attempt {attempt + 1}, retrying...")

            except requests.RequestException as e:
                if attempt == retries:
                    logger.error(f"Request failed after all retries: {str(e)}")
                    return None, f"Failed to fetch HTML: {str(e)}"
                logger.warning(
                    f"Request failed on attempt {attempt + 1}: {str(e)}, retrying..."
                )

            # Exponential backoff between retries
            if attempt < retries:
                sleep_time = backoff_factor * (2**attempt)
                time.sleep(sleep_time)


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


@log_elapsed_time
def fetch_single_article(url: str) -> Dict[str, str]:
    """
    Fetch and parse a single news article with multiple fallback approaches.

    Args:
        url (str): URL of the article to fetch
    Returns:
        Dict[str, str]: Article information including parsed content and raw HTML
    """
    # Initialize empty result with default values
    result = {
        "title": "Error",
        "content": "",
        "authors": "Unknown",
        "published_date": "Unknown",
        "origin": url,
        "status": "error",
        "html": "",
    }

    try:
        logger.info(f"Fetching article from URL: {url}")
        # First attempt: Fetch raw HTML with improved timeout handling
        raw_html, error = fetch_html(url)
        if error:
            logger.error(f"Failed to fetch HTML: {error}")
            result["content"] = error
            return result

        logger.info("Successfully fetched HTML content")

        # Clean HTML content
        cleaned_html = clean_html(raw_html)
        result["html"] = cleaned_html

        # First parsing attempt: Use NewsPlease
        try:
            logger.info(
                f"Attempting to parse with NewsPlease (timeout: {TIMEOUT.NEWSPLEASE_TIMEOUT}s)"
            )

            @timeout_decorator(TIMEOUT.NEWSPLEASE_TIMEOUT)
            def parse_with_newsplease(url):
                return NewsPlease.from_url(url)

            article = parse_with_newsplease(url)
            if article:
                article_dict = article.get_dict()

                # Update result with NewsPlease data
                if article_dict.get("title"):
                    logger.info("Successfully parsed article with NewsPlease")
                    result.update(
                        {
                            "title": article_dict.get("title", "").strip()
                            or "Untitled",
                            "content": article_dict.get("maintext", "").strip(),
                            "authors": ", ".join(article_dict.get("authors", []))
                            or "Unknown",
                            "published_date": article_dict.get(
                                "date_publish", "Unknown"
                            ),
                            "status": "success",
                        }
                    )
                    return result
        except TimeoutError as te:
            logger.error(
                f"NewsPlease parsing timed out after {TIMEOUT.NEWSPLEASE_TIMEOUT} seconds"
            )
            logger.info("Falling back to custom extractor")
        except Exception as np_error:
            logger.error(f"NewsPlease parsing failed: {str(np_error)}")
            logger.info("Falling back to custom extractor")

        # Second parsing attempt: Use custom extractor
        if cleaned_html:
            logger.info("Attempting to parse with custom extractor")
            extracted = extract_article_content(cleaned_html)
            if extracted["status"] == "success":
                logger.info("Successfully parsed article with custom extractor")
                result.update(extracted)
                return result

        # If we got here with no content, update error message
        if not result["content"]:
            logger.error("Failed to extract content with both parsing methods")
            result["content"] = "Failed to extract article content"

        return result

    except Exception as e:
        result["content"] = f"Error fetching article: {str(e)}"
        return result


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
    with ThreadPoolExecutor(max_workers=RETRY.MAX_WORKERS) as executor:
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
    with ThreadPoolExecutor(
        max_workers=min(RETRY.MAX_WORKERS, len(links_to_crawl))
    ) as executor:
        link_results = list(executor.map(fetch_single_article, links_to_crawl))

    # Add non-error results
    for article in link_results:
        if article["status"] == "success" and len(results) < max_articles:
            results.append(article)

    return results[:max_articles]


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder for datetime objects"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


if __name__ == "__main__":
    # mcp.run(transport="stdio")

    print(fetch_recursive("https://huyenchip.com/blog", depth=1, max_articles=5))
