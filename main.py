import json
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from newsplease import NewsPlease

mcp = FastMCP("article")

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


@mcp.tool()
def fetch_article(url: str) -> str:
    """
    Fetch and parse a news article from a given URL
    
    Args:
        url (str): The URL of the article to fetch
    Returns:
        str: Article text
    """
    try:
        article = NewsPlease.from_url(url)
        article_dict = article.get_dict()
        
        # Get main text content and title
        text = article_dict.get('maintext', '')
        title = article_dict.get('title', '')
        
        # Combine title and text
        full_text = f"{title}\n\n{text}"
        
        return full_text
    except Exception as e:
        return f"Error fetching article: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
    # print(fetch_article("https://www.lesswrong.com/posts/5yFj7C6NNc8GPdfNo/subskills-of-listening-to-wisdom"))
