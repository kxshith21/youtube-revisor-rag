"""Revision and quiz routers (LLM modes from scripts/generate.py)."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool

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
from ..schemas import QuizRequest, QuizResponse, ReviseRequest, ReviseResponse

router = APIRouter(
    tags=["revise"],
    dependencies=[
        Depends(verify_api_key),
        Depends(require_index),
        Depends(require_llm),
    ],
)


@router.post(
    "/revise",
    response_model=ReviseResponse,
    summary="Generate a revision note for a topic",
    description=(
        "Calls generate(mode='revise') and returns a structured revision note "
        "plus the cited sources. Requires an LLM and a populated index."
    ),
    responses=ERROR_RESPONSES,
)
@limiter.limit(ask_rate_limit)
async def revise(
    request: Request,
    body: ReviseRequest,
    state: Annotated[AppState, Depends(get_state)],
) -> ReviseResponse:
    """Return a revision note for one topic."""
    request.state.query_length = len(body.topic)
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(
                services.revise,
                state.retriever,
                state.llm,
                body.topic,
                body.domain,
            ),
            timeout=state.settings.ASK_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise ApiError(504, "timeout", "The revision request timed out.") from exc
    request.state.result_count = len(result.sources)
    return result


@router.post(
    "/quiz",
    response_model=QuizResponse,
    summary="Generate a quiz for a topic",
    description=(
        "Calls generate(mode='quiz'), parses its JSON question list, and returns "
        "questions with the sources they cite. Malformed LLM JSON is retried "
        "once inside generate.py; a second failure returns 502."
    ),
    responses=ERROR_RESPONSES,
)
@limiter.limit(ask_rate_limit)
async def quiz(
    request: Request,
    body: QuizRequest,
    state: Annotated[AppState, Depends(get_state)],
) -> QuizResponse:
    """Return parsed quiz questions for one topic."""
    request.state.query_length = len(body.topic)
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(
                services.quiz,
                state.retriever,
                state.llm,
                body.topic,
                body.domain,
                body.num_questions,
            ),
            timeout=state.settings.ASK_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise ApiError(504, "timeout", "The quiz request timed out.") from exc
    request.state.result_count = len(result.sources)
    return result
