"""Ask router: full RAG answers, including SSE streaming."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from .. import services
from ..deps import (
    ERROR_RESPONSES,
    ApiError,
    AppState,
    ask_rate_limit,
    get_state,
    limiter,
    require_index,
    require_llm,
    verify_api_key,
)
from ..schemas import AnswerOut, AskRequest

router = APIRouter(
    tags=["ask"],
    dependencies=[
        Depends(verify_api_key),
        Depends(require_index),
        Depends(require_llm),
    ],
)

SSE_RESPONSES: dict[int | str, dict[str, Any]] = {
    **ERROR_RESPONSES,
    200: {
        "description": (
            "Server-Sent Events: `token` events with {\"text\": ...}, then a "
            "`done` event with the final Answer JSON, or an `error` event."
        ),
        "content": {
            "text/event-stream": {
                "schema": {"type": "string"},
                "example": (
                    "event: token\n"
                    "data: {\"text\": \"Sliding window is \"}\n\n"
                    "event: done\n"
                    "data: {\"query\": \"...\", \"answer\": \"...\", \"sources\": []}\n\n"
                ),
            }
        },
    },
}


def _history(body: AskRequest) -> list[dict[str, str]]:
    """Serialize request history into the shape generate.py expects."""
    return [item.model_dump() for item in body.history]


@router.post(
    "/ask",
    response_model=AnswerOut,
    summary="Ask a question (full RAG answer)",
    description=(
        "Retrieves context and calls the LLM. When no relevant chunks are found "
        "the response is still 200, with found=false and no LLM call. Responses "
        "for repeated cacheable questions include X-Cache: HIT."
    ),
    responses=ERROR_RESPONSES,
)
@limiter.limit(ask_rate_limit)
async def ask(
    request: Request,
    response: Response,
    body: AskRequest,
    state: Annotated[AppState, Depends(get_state)],
) -> AnswerOut:
    """Answer one question with validated citations."""
    request.state.query_length = len(body.query)
    try:
        answer, cache_status, result_count = await asyncio.wait_for(
            run_in_threadpool(
                services.ask_with_cache,
                state.retriever,
                state.llm,
                body.query,
                body.domain,
                body.top_k,
                _history(body),
                state.answer_cache,
            ),
            timeout=state.settings.ASK_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise ApiError(
            504,
            "timeout",
            "The answer request timed out. Try a shorter or simpler question.",
        ) from exc

    response.headers["X-Cache"] = cache_status
    request.state.result_count = result_count
    return answer


@router.post(
    "/ask/stream",
    summary="Ask a question (SSE stream)",
    description=(
        "Streams the answer as Server-Sent Events. Tokens arrive as "
        "`event: token` frames; the last frame is `event: done` with the final "
        "Answer JSON and validated sources. A comment heartbeat is sent every "
        "15 seconds. The stream stops if the client disconnects."
    ),
    responses=SSE_RESPONSES,
)
@limiter.limit(ask_rate_limit)
async def ask_stream(
    request: Request,
    body: AskRequest,
    state: Annotated[AppState, Depends(get_state)],
) -> StreamingResponse:
    """Stream tokens for one question, ending with the validated Answer."""
    request.state.query_length = len(body.query)
    session = services.AnswerStream(
        state.retriever,
        state.llm,
        body.query,
        body.domain,
        body.top_k,
        _history(body),
    )
    events = services.sse_events(
        session,
        heartbeat_s=state.settings.STREAM_HEARTBEAT_S,
        timeout_s=state.settings.STREAM_TIMEOUT_S,
        request_id=getattr(request.state, "request_id", ""),
    )
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
