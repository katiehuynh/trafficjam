"""API-facing request/response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    max_tokens: int = Field(default=32, ge=1, le=512)
    model: str = Field(default="large", description="'large', 'small', or 'auto'")
    priority: int | None = Field(default=None, ge=0, le=2, description="0 highest; defaults to tenant tier")
    stream: bool = False
    allow_fallback: bool = True


class GenerateResponse(BaseModel):
    id: str
    tenant_id: str
    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    e2e_ms: float
    preemptions: int
    fell_back: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""


class ChaosKillRequest(BaseModel):
    count: int = Field(default=1, ge=1, le=10)
    profile: str | None = Field(default=None, description="'large' | 'small' | null for any")


class ScaleRequest(BaseModel):
    profile: str
    count: int = Field(ge=0, le=16)
