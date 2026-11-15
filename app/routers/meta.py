"""Health and index metadata endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from .. import services
from ..deps import (
    ERROR_RESPONSES,
    AppState,
    get_state,
    require_index,
    verify_api_key,
)
from ..schemas import Domain, HealthOut, TopicsResponse, VideosResponse

router = APIRouter(tags=["meta"])


@router.get(
    "/health",
    response_model=HealthOut,
    summary="Service health",
    description=(
        "Always returns 200. status is 'degraded' when Qdrant, the collection, "
        "the embedding model, or the LLM is unavailable. Never contains secrets."
    ),
)
def health(state: Annotated[AppState, Depends(get_state)]) -> HealthOut:
    """Report readiness without exposing credentials."""
    return services.health(state)


@router.get(
    "/topics",
    response_model=TopicsResponse,
    summary="Available domains",
    description=(
        "Per-domain chunk and distinct video counts from Qdrant. Cached in "
        "memory for 10 minutes."
    ),
    responses=ERROR_RESPONSES,
    dependencies=[Depends(verify_api_key), Depends(require_index)],
)
def topics(state: Annotated[AppState, Depends(get_state)]) -> TopicsResponse:
    """Return domain counts for the indexed collection."""
    items = services.topics(state.retriever, state.topics_cache)
    return TopicsResponse(topics=items, count=len(items))


@router.get(
    "/videos",
    response_model=VideosResponse,
    summary="Indexed videos",
    description=(
        "Distinct videos with chunk counts, optionally filtered by domain and a "
        "case-insensitive title substring. Results are paginated and cached in "
        "memory for 10 minutes."
    ),
    responses=ERROR_RESPONSES,
    dependencies=[Depends(verify_api_key), Depends(require_index)],
)
def videos(
    state: Annotated[AppState, Depends(get_state)],
    domain: Annotated[Domain | None, Query(description="Filter by domain.")] = None,
    q: Annotated[
        str | None,
        Query(max_length=100, description="Case-insensitive title substring."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> VideosResponse:
    """Return a page of distinct indexed videos."""
    title_query = q.strip() if q else None
    page, total = services.videos(
        state.retriever,
        state.videos_cache,
        domain,
        title_query or None,
        limit,
        offset,
    )
    return VideosResponse(
        videos=page,
        count=len(page),
        total=total,
        limit=limit,
        offset=offset,
    )
