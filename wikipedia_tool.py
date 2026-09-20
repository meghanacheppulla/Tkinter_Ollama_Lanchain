"""Wikipedia lookup exposed as a predefined LangChain tool."""

from __future__ import annotations

import wikipedia.wikipedia as wikipedia_api
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# The wikipedia package defaults to HTTP, which can return an HTML redirect
# instead of JSON on some networks.
wikipedia_api.API_URL = "https://en.wikipedia.org/w/api.php"
wikipedia_api.set_user_agent(
    "CampusCompass/1.0 (educational desktop application)"
)


def create_wikipedia_tool() -> WikipediaQueryRun:
    """Create a bounded Wikipedia tool suitable for desktop use."""
    wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=4_000)
    return WikipediaQueryRun(api_wrapper=wrapper)
