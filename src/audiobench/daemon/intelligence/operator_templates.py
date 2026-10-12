from typing import Any, ClassVar, Literal, TypedDict

from pydantic import BaseModel, Field


class DaemonProposal(TypedDict):
    region_id: str
    operator_template: str
    parameters: dict[str, Any]
    schema_version: int


class OperatorPayload(BaseModel):
    region_id: str
    operator_template: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = 2


class KnowledgeGapPayload(BaseModel):
    distance: float
    threshold: float
    window_days: int = 14
    diagnostic: str
    recommended_ingestion: str
    stranded_queries: list[str] = Field(default_factory=list)
    name_source: str = "legacy_unnamed"
    legacy_backfilled: bool = False


class DaemonProposalEnvelope(BaseModel):
    proposal_type: Literal["operator", "knowledge_gap"]
    schema_version: int = 2
    title: str
    summary: str
    evidence: str
    payload: dict[str, Any] = Field(default_factory=dict)


class LectureModeOperator(BaseModel):
    SCHEMA_VERSION: ClassVar[int] = 2
    frequency_multiplier: float = Field(ge=0.1, le=10.0)

_TEMPLATE_CLASSES: dict[str, type[BaseModel]] = {
    "LectureModeOperator": LectureModeOperator,
}

