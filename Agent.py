"""
The agent loop: system prompt, LLM backend selection, and the tool-calling
loop itself — including two safeguards added after a real failure mode was
observed in testing:

  1. If no resume is loaded, refuse to call the LLM at all. Previously, an
     unloaded resume plus an ambiguous message (e.g. a stray filename typed
     as a chat message) let the model improvise a plausible-sounding but
     entirely fabricated answer instead of a clear error.

  2. If the model's response contains no real tool_calls but its text
     *looks* like a fabricated tool call (some models — even ones with
     documented tool-calling support — occasionally narrate a tool call in
     prose instead of emitting one), reject it and force one corrective
     retry rather than returning the hallucinated text as a final answer.
"""

import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

import Config as config
import Resume_index as resume_index
from Feedback import check_past_corrections
from Tool import TOOLS, TOOL_REGISTRY

SYSTEM_PROMPT = """
You are an HR Assistant helping recruiters review a candidate's resume.

Rules:
- Always call search_resume to find relevant content before answering —
  never answer about the candidate's skills, projects, education, or
  experience from assumption.
- NEVER write out a tool call or its result as plain text. If you need to
  search the resume, you must issue a real tool call, not describe one.
- The retrieved chunks include similarity scores. Trust chunks with higher
  scores more; if the top result still seems weakly related, say the resume
  may not clearly cover that topic rather than overstating confidence.
- If search_resume returns nothing relevant, say clearly that the resume
  doesn't contain that information — do not guess or fill in gaps.
- Once you have enough information, answer directly. Do not call the tool
  again if you already have what you need.
""".strip()

# Heuristic pattern for spotting a model narrating a fake tool call/result
# in plain text instead of issuing a real one (e.g. `{"name": "search_resume"`
# or `"output": {...}` appearing in the response content).
_FAKE_TOOL_CALL_PATTERN = re.compile(
    r'["\']name["\']\s*:\s*["\']search_resume["\']|["\']output["\']\s*:\s*\{',
    re.IGNORECASE,
)


def _init_llm():
    """
    Returns the base chat model, before tools are bound.

    Groq and Ollama use their dedicated LangChain integration classes
    directly rather than init_chat_model(..., model_provider=...) — some
    LangChain versions raise NotImplementedError on bind_tools() when
    routed through init_chat_model for less-common providers.
    """
    if config.BACKEND == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=config.GROQ_MODEL_NAME, temperature=0.0)
    elif config.BACKEND == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.OLLAMA_MODEL_NAME, temperature=0.0)
    else:
        return init_chat_model(config.OPENAI_MODEL_NAME, model_provider="openai", temperature=0.0)


def _looks_like_fake_tool_call(text: str) -> bool:
    return bool(text) and bool(_FAKE_TOOL_CALL_PATTERN.search(text))


def run_hr_assistant(user_message: str, max_tool_calls: int = config.MAX_TOOL_CALLS) -> str:
    # --- Guard #1: don't even call the LLM if there's nothing to search ---
    if not resume_index.is_resume_loaded():
        return ("No resume is currently loaded. Use the 'load <filename>' "
                "command first (file must be in the uploads/ folder).")

    llm = _init_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    correction_hint = check_past_corrections(user_message)
    if correction_hint:
        messages.append(SystemMessage(content=correction_hint))

    messages.append(HumanMessage(content=user_message))

    tool_call_count = 0
    already_retried_fake_call = False

    while True:
        if tool_call_count >= max_tool_calls:
            messages.append(HumanMessage(
                content="Tool call limit reached. Answer now using only what you've already found."
            ))
            final = llm.invoke(messages)
            return final.content

        response = llm_with_tools.invoke(messages)

        # --- Guard #2: catch a model narrating a fake tool call as text ---
        if not response.tool_calls and _looks_like_fake_tool_call(response.content):
            print("[warning] Model appears to have fabricated a tool call in "
                  "plain text instead of issuing a real one.")
            if already_retried_fake_call:
                return ("I wasn't able to properly search the resume due to a "
                        "tool-calling issue with the current model backend. "
                        "Try again, or switch BACKEND in config.py.")
            already_retried_fake_call = True
            messages.append(HumanMessage(
                content="Your previous response described a tool call instead of "
                        "issuing a real one. Issue an actual tool call now — do not "
                        "write out JSON or describe what the tool would return."
            ))
            continue

        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            tool_call_count += 1
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            tool_fn = TOOL_REGISTRY.get(tool_name)
            result = tool_fn.invoke(tool_args) if tool_fn else f"Unknown tool: {tool_name}"

            messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))