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


@dataclass
class ApplicationConfig:
    """Application-wide configuration settings"""

    # Timeout settings
    CONNECT_TIMEOUT: int = 5  # seconds for initial connection
    READ_TIMEOUT: int = 10  # seconds for reading response
    NEWSPLEASE_TIMEOUT: int = 10  # seconds for NewsPlease parsing
    TOTAL_REQUEST_TIMEOUT: int = 15  # maximum time for entire request

    # Retry settings
    MAX_RETRIES: int = 2
    BACKOFF_FACTOR: float = 0.3
    MAX_WORKERS: int = 10  # for ThreadPoolExecutor

    # HTTP settings
    USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    CHUNK_SIZE: int = 8192
    ALLOW_REDIRECTS: bool = True

    # Batch settings
    ARTICLES_BATCH_SIZE: int = 10  # Default batch size for fetch_articles
    RECURSIVE_BATCH_SIZE: int = 10  # Default batch size for fetch_recursive

    # Parsing settings
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
CONFIG = ApplicationConfig()

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
        for tag in soup.find_all(CONFIG.UNNECESSARY_TAGS):
            tag.decompose()

        # Remove common ad-related elements
        for element in soup.find_all(class_=re.compile(CONFIG.AD_PATTERN)):
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
                if key not in CONFIG.UNWANTED_ATTRIBUTES
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
            class_=re.compile(CONFIG.ARTICLE_PATTERN)
        )
        if main_content:
            # Remove unwanted elements
            for tag in main_content.find_all(CONFIG.UNNECESSARY_TAGS):
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
    total_timeout: int = CONFIG.TOTAL_REQUEST_TIMEOUT,
    connect_timeout: int = CONFIG.CONNECT_TIMEOUT,
    retries: int = CONFIG.MAX_RETRIES,
    backoff_factor: float = CONFIG.BACKOFF_FACTOR,
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
    headers = {"User-Agent": CONFIG.USER_AGENT}

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
                    allow_redirects=CONFIG.ALLOW_REDIRECTS,
                ) as response:
                    response.raise_for_status()
                    content = []

                    total_chunks = 0
                    total_size = 0
                    start_time = time.time()

                    # Read the content in chunks with total timeout enforcement
                    for chunk in response.iter_content(
                        chunk_size=CONFIG.CHUNK_SIZE, decode_unicode=True
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
                f"Attempting to parse with NewsPlease (timeout: {CONFIG.NEWSPLEASE_TIMEOUT}s)"
            )

            @timeout_decorator(CONFIG.NEWSPLEASE_TIMEOUT)
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
                f"NewsPlease parsing timed out after {CONFIG.NEWSPLEASE_TIMEOUT} seconds"
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
def fetch_articles(
    urls: List[str], batch_size: int = CONFIG.ARTICLES_BATCH_SIZE
) -> List[Dict[str, str]]:
    """
    Fetch and parse multiple news articles in parallel using ThreadPoolExecutor with batching

    Args:
        urls (List[str]): List of URLs to fetch
        batch_size (int): Number of articles to process in parallel per batch (default: 5)
    Returns:
        List[Dict[str, str]]: List of article information dictionaries
    """
    results = []
    total_urls = len(urls)
    processed = 0

    # Process URLs in batches to prevent memory overload
    for i in range(0, total_urls, batch_size):
        batch = urls[i : min(i + batch_size, total_urls)]

        with ThreadPoolExecutor(
            max_workers=min(CONFIG.MAX_WORKERS, len(batch))
        ) as executor:
            futures = {executor.submit(fetch_single_article, url): url for url in batch}

            for future in futures:
                try:
                    result = future.result(timeout=CONFIG.TOTAL_REQUEST_TIMEOUT)
                    results.append(result)
                    processed += 1
                    logger.info(
                        f"Processed {processed}/{total_urls} articles ({(processed/total_urls)*100:.1f}%)"
                    )
                except Exception as e:
                    url = futures[future]
                    logger.error(f"Error processing {url}: {str(e)}")
                    results.append(
                        {
                            "title": "Error",
                            "content": f"Failed to fetch: {str(e)}",
                            "origin": url,
                            "status": "error",
                        }
                    )
                    processed += 1

    return results


@mcp.tool()
def fetch_recursive(
    url: str,
    depth: int = 1,
    max_articles: int = 5,
    batch_size: int = CONFIG.RECURSIVE_BATCH_SIZE,
) -> List[Dict[str, str]]:
    """
    Recursively fetch articles from a website up to specified depth with parallel processing.

    Args:
        url (str): The starting URL to fetch
        depth (int): How many levels deep to crawl (default: 1)
        max_articles (int): Maximum number of articles to fetch (default: 5)
        batch_size (int): Number of articles to process in parallel per batch (default: 3)
    Returns:
        List[Dict[str, str]]: List of article information dictionaries
    """
    if depth < 1 or max_articles < 1:
        return [{"error": "Depth and max_articles must be greater than 0"}]

    visited_urls = set()
    results = []
    urls_to_process = [(url, 0)]  # (url, current_depth)

    while urls_to_process and len(results) < max_articles:
        current_batch = []
        batch_depths = []

        # Create batch of URLs to process
        while urls_to_process and len(current_batch) < batch_size:
            current_url, current_depth = urls_to_process.pop(0)
            if current_url not in visited_urls:
                current_batch.append(current_url)
                batch_depths.append(current_depth)
                visited_urls.add(current_url)

        if not current_batch:
            break

        # Process current batch in parallel
        with ThreadPoolExecutor(
            max_workers=min(CONFIG.MAX_WORKERS, len(current_batch))
        ) as executor:
            futures = {
                executor.submit(fetch_single_article, url): (url, depth)
                for url, depth in zip(current_batch, batch_depths)
            }

            for future in futures:
                try:
                    article = future.result(timeout=CONFIG.TOTAL_REQUEST_TIMEOUT)
                    url, current_depth = futures[future]

                    if article["status"] == "success":
                        results.append(article)
                        logger.info(
                            f"Fetched article {len(results)}/{max_articles} from depth {current_depth}"
                        )

                        # If we can go deeper, extract and add links to process
                        if current_depth < depth - 1:
                            raw_html, _ = fetch_html(url)
                            if raw_html:
                                cleaned_html = clean_html(raw_html)
                                new_links = extract_links(cleaned_html, url)

                                # Add new links with incremented depth
                                for link in new_links:
                                    if link not in visited_urls:
                                        urls_to_process.append(
                                            (link, current_depth + 1)
                                        )

                except Exception as e:
                    url, _ = futures[future]
                    logger.error(f"Error processing {url}: {str(e)}")

                if len(results) >= max_articles:
                    break

    return results[:max_articles]


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder for datetime objects"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


if __name__ == "__main__":
    mcp.run(transport="stdio")

    # run fetch_articles with multiple urls:
    # print(
    #     fetch_articles(
    #         [
    #             "https://huyenchip.com/2025/01/16/ai-engineering-pitfalls.html",
    #             "https://huyenchip.com/2025/01/07/agents.html",
    #             "https://huyenchip.com/2024/07/25/genai-platform.html",
    #             "https://huyenchip.com/2024/04/17/personal-growth.html",
    #             "https://huyenchip.com/2024/03/14/ai-oss.html",
    #             "https://huyenchip.com/2024/02/28/predictive-human-preference.html",
    #             "https://huyenchip.com/2024/01/16/sampling.html",
    #             "https://huyenchip.com/2023/10/10/multimodal.html",
    #             "https://huyenchip.com/2023/08/16/llm-research-open-challenges.html",
    #             "https://huyenchip.com/2023/06/07/generative-ai-strategy.html",
    #             "https://huyenchip.com/2023/05/02/rlhf.html",
    #             "https://huyenchip.com/2023/04/11/llm-engineering.html",
    #             "https://huyenchip.com/2023/01/24/what-we-look-for-in-a-candidate.html",
    #             "https://huyenchip.com/2023/01/08/self-serve-feature-platforms.html",
    #             "https://huyenchip.com/2022/12/27/books-for-every-engineer.html",
    #             "https://huyenchip.com/2022/08/03/stream-processing-for-data-scientists.html",
    #             "https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html",
    #             "https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html",
    #             "https://huyenchip.com/2021/09/13/data-science-infrastructure.html",
    #             "https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html",
    #             "https://huyenchip.com/2021/02/27/why-not-join-a-startup.html",
    #             "https://huyenchip.com/2020/12/30/mlops-v2.html",
    #             "https://huyenchip.com/2020/12/27/real-time-machine-learning.html",
    #             "https://huyenchip.com/2020/10/27/ml-systems-design-stanford.html",
    #             "https://huyenchip.com/2020/06/22/mlops.html",
    #             "https://huyenchip.com/2020/01/18/tech-workers-19k-compensation-details.html",
    #             "https://huyenchip.com/2019/12/28/books-that-shaped-my-decade.html",
    #             "https://huyenchip.com/2019/12/23/leaving-nvidia-lessons.html",
    #             "https://huyenchip.com/2019/12/18/key-trends-neurips-2019.html",
    #             "https://huyenchip.com/2019/08/21/glassdoor-interview-reviews-tech-hiring-cultures.html",
    #             "https://huyenchip.com/2019/08/05/free-online-machine-learning-curriculum.html",
    #             "https://huyenchip.com/2019/07/21/machine-learning-interviews.html",
    #             "https://huyenchip.com/2019/05/12/top-8-trends-from-iclr-2019.html",
    #             "https://huyenchip.com/2019/03/11/silicon-valley-misogyny.html",
    #             "https://huyenchip.com/2018/11/16/building-meaningful-relationships.html",
    #             "https://huyenchip.com/2018/10/08/career-advice-recent-cs-graduates.html",
    #             "https://huyenchip.com/2018/10/04/sotawhat.html",
    #             "https://huyenchip.com/2018/03/30/guide-to-Artificial-Intelligence-Stanford.html",
    #             "https://huyenchip.com/2017/07/28/confession.html",
    #         ]
    #     )
    # )
