# Article Analyzer MCP Server

A Model Context Protocol (MCP) server that fetches article content from URLs and provides it as context for LLM models through Cline.

## Purpose

This MCP server enables LLMs to:

- Fetch and read article content from any URL
- Get cleaned and parsed article text (limited to 1000 words)
- Use article content as context for analysis and summarization

## Setup with Cline

1. Clone and set up the repository:

```bash
git clone https://github.com/yourusername/article-analyzer.git
cd article-analyzer
uv venv
uv pip install -r requirements.txt
```

2. Configure the MCP server in your Cline settings:

Add the following configuration to your Cline MCP server settings

```json
{
  "mcpServers": {
    "article": {
      "command": "/ABSOLUTE/PATH/TO/UV" // Path to the uv executable
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/PARENT/FOLDER/article-analyzer",
        "run",
        "main.py"
      ]
    }
  }
}
```

Note: Adjust the paths in the configuration to match your system.

## Usage Example

Example prompt for LLM models:

```python
{
    "text": "Analyze the article at https://www.seangoedecke.com/how-i-use-llms/ and summarize it."
}
```

## Features

- Fast article fetching using news-please library
- Automatic text cleaning and parsing
- Integration with Cline/ClaudeDesktop for LLM context
- Support for various news and article websites

## Dependencies

- Python >=3.8
- mcp[cli] - For MCP server functionality
- news-please - For article fetching and parsing
- httpx - For HTTP requests

## License

[MIT License included in repository]
