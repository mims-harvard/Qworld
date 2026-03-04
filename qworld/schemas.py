"""Structured output schemas for different providers."""
from typing import List, Optional
from pydantic import BaseModel, Field


class Scenario(BaseModel):
    scenario_name: str = Field(description="Brief name for the scenario")
    scenario_description: str = Field(description="Detailed description of the context")


class ScenariosResponse(BaseModel):
    scenarios: List[Scenario] = Field(description="List of scenarios")


class Perspective(BaseModel):
    perspective_name: str = Field(description="Brief name for the perspective")
    perspective_description: str = Field(description="What aspects this perspective evaluates")


class PerspectivesResponse(BaseModel):
    perspectives: List[Perspective] = Field(description="List of perspectives")


class Criterion(BaseModel):
    criterion: str = Field(description="Specific evaluation criterion")
    points: int = Field(description="Importance weight")
    reasoning: str = Field(description="Reasoning for the criterion")


class CriteriaResponse(BaseModel):
    criteria: List[Criterion] = Field(description="List of criteria")


class PolarityItem(BaseModel):
    positive: bool = Field(description="True if meeting criterion is good")
    reasoning: str = Field(description="Reasoning for the criterion")


class PolarityResponse(BaseModel):
    criteria: List[PolarityItem] = Field(description="Polarity for each criterion")


# Alignment evaluation schemas
class ExpertCriterionCoverage(BaseModel):
    criterion: str = Field(description="Original expert criterion text")
    is_covered: str = Field(description="'yes' or 'no'")
    comment: str = Field(description="Brief explanation")


class ExpertCoverageResponse(BaseModel):
    expert_criteria: List[ExpertCriterionCoverage] = Field(description="Coverage results for expert criteria")


class ModelCriterionAlignment(BaseModel):
    criterion: str = Field(description="Model criterion text")
    is_covered: str = Field(description="'yes' or 'no'")
    is_valuable: str = Field(description="'yes' or 'no'")
    reason: str = Field(description="Brief explanation")


class ModelAlignmentResponse(BaseModel):
    model_criteria: List[ModelCriterionAlignment] = Field(description="Alignment results for model criteria")


# Map agent names to response schemas
AGENT_SCHEMAS = {
    "ScenarioAnalyzer": ScenariosResponse,
    "ScenarioExpander": ScenariosResponse,
    "PerspectiveAnalyzer": PerspectivesResponse,
    "PerspectiveExpander": PerspectivesResponse,
    "PerspectiveReviewer": PerspectivesResponse,
    "CriteriaGenerator": CriteriaResponse,
    "CriteriaExpander": CriteriaResponse,
    "CriteriaReviewer": CriteriaResponse,
    "NegativeCriteriaChecker": PolarityResponse,
    "CriteriaScoreAssigner": CriteriaResponse,
    "CriteriaAlignmentJudger": ExpertCoverageResponse,
    "ModelUniqueCriteriaJudger": ModelAlignmentResponse,
}
