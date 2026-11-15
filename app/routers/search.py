"""Search router: retrieval only, no LLM."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from .. import services
from ..deps import (
    ERROR_RESPONSES,
    AppState,
    get_state,
    limiter,
    require_index,
    search_rate_limit,
    verify_api_key,
)
from ..schemas import SearchRequest, SearchResponse, SearchResultOut

router = APIRouter(
    tags=["search"],
    dependencies=[Depends(verify_api_key), Depends(require_index)],
)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search the indexed videos",
    description=(
        "Embeds the query and searches Qdrant only. No LLM call is made. "
        "An empty results list is a normal 200 response."
    ),
    responses=ERROR_RESPONSES,
)
@limiter.limit(search_rate_limit)
def search(
    request: Request,
    body: SearchRequest,
    state: Annotated[AppState, Depends(get_state)],
) -> SearchResponse:
    """Retrieve relevant transcript chunks for a query."""
    request.state.query_length = len(body.query)
    results, latency_ms = services.search(
        state.retriever,
        body.query,
        body.domain,
        body.top_k,
        body.use_mmr,
        body.use_reranker,
    )
    request.state.result_count = len(results)
    return SearchResponse(
        query=body.query,
        results=[SearchResultOut.model_validate(r.to_dict()) for r in results],
        count=len(results),
        latency_ms=round(latency_ms, 2),
    )
