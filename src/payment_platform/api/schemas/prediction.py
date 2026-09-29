from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    transaction_id: str


class PredictionResponse(BaseModel):
    transaction_id: str
    failure_probability: float = Field(ge=0, le=1)
    success_probability: float = Field(ge=0, le=1)
    model_version: str