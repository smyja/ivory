from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from data.dataset_duckdb import execute_query, preview_schema
from labels.sqlite_store import upsert_label_by_text, upsert_label_by_row_id


router = APIRouter(prefix="/query", tags=["query"])


class WhereCond(BaseModel):
    column: str
    op: str
    value: Any


class OrderBy(BaseModel):
    column: str
    direction: str = Field("asc", pattern="^(?i)(asc|desc)$")


class QuerySpec(BaseModel):
    dataset: str
    select: Optional[List[str]] = None
    where: Optional[List[WhereCond]] = None
    order_by: Optional[OrderBy] = None
    limit: Optional[int] = 100
    offset: Optional[int] = 0
    return_total: Optional[bool] = True


@router.post("/run")
def run_query(spec: QuerySpec) -> Dict[str, Any]:
    try:
        result = execute_query(
            dataset_name=spec.dataset,
            select=spec.select,
            where=[w.model_dump() for w in (spec.where or [])],
            order_by=spec.order_by.model_dump() if spec.order_by else None,
            limit=spec.limit,
            offset=spec.offset,
            return_total=bool(spec.return_total),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query failed: {e}")


@router.get("/preview/{dataset}")
def preview(dataset: str) -> Dict[str, Any]:
    try:
        return preview_schema(dataset)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preview failed: {e}")


class LabelUpsert(BaseModel):
    dataset: str
    label: str
    text: str
    value: str


@router.post("/label/upsert")
def upsert_label(payload: LabelUpsert) -> Dict[str, Any]:
    try:
        upsert_label_by_text(payload.dataset, payload.label, payload.text, payload.value)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Label upsert failed: {e}")


class LabelUpsertByRowId(BaseModel):
    dataset: str
    label: str
    row_id: str
    value: str
    text: Optional[str] = None


@router.post("/label/upsert_row")
def upsert_label_row(payload: LabelUpsertByRowId) -> Dict[str, Any]:
    try:
        upsert_label_by_row_id(
            dataset_name=payload.dataset,
            label_name=payload.label,
            row_id=payload.row_id,
            value=payload.value,
            text=payload.text,
        )
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Label upsert (row) failed: {e}")
