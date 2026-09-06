"""LangChain tool definitions available to the agent."""

from langchain_core.tools import tool

import Resume_index as resume_index


@tool
def search_resume(query: str) -> str:
    """Semantically search the currently loaded resume for content relevant
    to the query (skills, projects, education, or experience). Returns the
    most relevant sentence-level chunks from the actual uploaded document,
    ranked by embedding similarity rather than exact keyword overlap — so it
    can match questions phrased differently than the resume's wording."""
    return resume_index.search(query)


TOOLS = [search_resume]
TOOL_REGISTRY = {t.name: t for t in TOOLS}