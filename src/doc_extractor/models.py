from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class FieldSpec(BaseModel):
    name: str
    description: str
    type: Literal["string", "number", "integer", "boolean", "list"] = "string"


class OutputSchema(BaseModel):
    schema_id: str
    fields: list[FieldSpec]


class Document(BaseModel):
    path: str
    text: str
    pages: list[str]
    n_pages: int


class FieldHit(BaseModel):
    name: str
    value: str | None = None
    grounding: str | None = None


class SchemaExtraction(BaseModel):
    fields: list[FieldHit]


class ValueGroup(BaseModel):
    canonical_value: str | None
    count: int
    variants: list[str]
    groundings: list[str]


class ExtractedField(BaseModel):
    name: str
    value: str | None = None
    count: int
    total_calls: int
    score: float
    candidates: list[ValueGroup]


class ExtractionResult(BaseModel):
    schema_id: str
    document: str
    total_calls: int
    fields: list[ExtractedField]


class MergeGroup(BaseModel):
    canonical_value: str
    variants: list[str]


class MergeGroups(BaseModel):
    groups: list[MergeGroup]
