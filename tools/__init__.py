"""
Web search and external tools for the RAG chatbot.

This package provides web search capabilities using Tavily API.
"""

from .web_search import search_web, format_search_results, should_use_web_search

__all__ = ["search_web", "format_search_results", "should_use_web_search"]