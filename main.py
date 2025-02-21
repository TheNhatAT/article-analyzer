from mcp.server.fastmcp import FastMCP
from newsplease import NewsPlease

mcp = FastMCP("article")

@mcp.tool()
def fetch_article(url: str) -> dict:
    """
    Fetch and parse a news article from a given URL
    
    Args:
        url (str): The URL of the article to fetch
    """
    try:
        article = NewsPlease.from_url(url)
        return article.get_dict()
    except Exception as e:
        return f"Error fetching article: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
