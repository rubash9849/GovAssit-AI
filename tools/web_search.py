import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using Tavily API.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, content, and URL
    """
    try:
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        # Initialize Tavily client
        client = TavilyClient(api_key=api_key)
        
        # Perform search
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",  # or "advanced" for more thorough search
            include_answer=True,    # Get AI-generated answer
            include_raw_content=False
        )
        
        results = []
        
        # Add AI answer if available
        if response.get("answer"):
            results.append({
                "type": "answer",
                "content": response["answer"],
                "source": "Tavily AI Summary"
            })
        
        # Add search results
        for result in response.get("results", []):
            results.append({
                "type": "search_result",
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "url": result.get("url", ""),
                "score": result.get("score", 0.0)
            })
        
        return results
        
    except ImportError:
        print("Error: Tavily package not installed. Install with: pip install tavily-python")
        return []
    except Exception as e:
        print(f"Error performing web search: {e}")
        return []


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Format search results into a readable string for the LLM.
    
    Args:
        results: List of search result dictionaries
        
    Returns:
        Formatted string of search results
    """
    if not results:
        return "No search results found."
    
    formatted = []
    
    for i, result in enumerate(results, 1):
        if result["type"] == "answer":
            formatted.append(f"AI Summary:\n{result['content']}\n")
        else:
            formatted.append(
                f"[{i}] {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Content: {result['content']}\n"
            )
    
    return "\n".join(formatted)


def should_use_web_search(query: str) -> bool:
    """
    Determine if a query should trigger web search.
    
    Args:
        query: User query string
        
    Returns:
        True if web search should be used, False otherwise
    """
    # Keywords that indicate web search need
    web_search_indicators = [
        "latest", "recent", "current", "today", "news", "update",
        "what's happening", "what happened", "breaking",
        "2024", "2025", "this year", "this month"
    ]
    
    query_lower = query.lower()
    
    # Check for web search indicators
    for indicator in web_search_indicators:
        if indicator in query_lower:
            return True
    
    return False