"""FastAPI service wrapping the ADK agent, with guardrails and structured logging.

POST /chat  {"message": "...", "session_id": "optional"}  ->  {"reply": ..., "session_id": ...}
GET  /healthz
"""
import asyncio
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

from app.agent import root_agent
from app.guardrails import check_input, check_output
from app.observability import get_logger, log_event, trace_id_from_header

APP_NAME = "market-risk-assistant"
AGENT_TIMEOUT_S = 60

app = FastAPI(title="Market Risk Assistant", version="1.0.0")
logger = get_logger()
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=5000)
    session_id: str | None = None
    user_id: str = "anonymous"


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    blocked: bool = False
    latency_ms: int


async def run_agent(user_id: str, session_id: str, text: str) -> tuple[str, dict]:
    """Run the agent graph; return (final_text, token_usage)."""
    existing = await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    if existing is None:
        await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    message = types.Content(role="user", parts=[types.Part(text=text)])
    final_text, prompt_tokens, output_tokens, agents_used = "", 0, 0, []
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
        usage = getattr(event, "usage_metadata", None)
        if usage:
            prompt_tokens += getattr(usage, "prompt_token_count", 0) or 0
            output_tokens += getattr(usage, "candidates_token_count", 0) or 0
        author = getattr(event, "author", None)
        if author and author not in agents_used and author != "user":
            agents_used.append(author)
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)
    return final_text, {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
                        "agents": agents_used}


async def run_agent_with_retry(user_id: str, session_id: str, text: str):
    delay = 2
    for attempt in range(4):
        try:
            return await run_agent(user_id, session_id, text)
        except genai_errors.ServerError as exc:  # 5xx such as 503 high demand
            if attempt == 3:
                raise
            log_event(logger, "agent_retry", logging.WARNING, session_id=session_id,
                      attempt=attempt + 1, error=str(exc)[:120])
            await asyncio.sleep(delay)
            delay *= 2


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    start = time.perf_counter()
    trace = trace_id_from_header(request.headers.get("x-cloud-trace-context"))
    session_id = req.session_id or str(uuid.uuid4())
    ms = lambda: int((time.perf_counter() - start) * 1000)   # noqa: E731

    gi = check_input(req.message)
    log_event(logger, "request_received", trace_id=trace, session_id=session_id,
              input_chars=len(req.message), pii_found=gi.pii_found)
    if not gi.allowed:
        log_event(logger, "guardrail_blocked", logging.WARNING, trace_id=trace, session_id=session_id,
                  stage="input", reasons=gi.reasons)
        return ChatResponse(reply=gi.user_message, session_id=session_id, blocked=True, latency_ms=ms())

    try:
        text, usage = await asyncio.wait_for(run_agent_with_retry(req.user_id, session_id, gi.text), AGENT_TIMEOUT_S)
    except Exception as exc:
        log_event(logger, "agent_error", logging.ERROR, trace_id=trace, session_id=session_id,
                  error_type=type(exc).__name__, error=str(exc)[:300], latency_ms=ms())
        raise HTTPException(status_code=502, detail="The assistant is unavailable right now. Please retry.")

    go = check_output(text)
    if not go.allowed:
        log_event(logger, "guardrail_blocked", logging.WARNING, trace_id=trace, session_id=session_id,
                  stage="output", reasons=go.reasons)
        return ChatResponse(reply=go.user_message, session_id=session_id, blocked=True, latency_ms=ms())

    log_event(logger, "agent_response", trace_id=trace, session_id=session_id, latency_ms=ms(),
              output_chars=len(go.text), output_flags=go.reasons, **usage)
    return ChatResponse(reply=go.text, session_id=session_id, latency_ms=ms())
