from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    amount: float = Field(gt=0)
    merchant_category: str
    payment_method: str
    user_segment: str
    device_type: str
    network_quality: str
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    retry_count: int = Field(ge=0)
    transaction_velocity: int = Field(ge=0)
    user_method_success_rate: float = Field(ge=0, le=1)
    merchant_method_success_rate: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    failure_probability: float = Field(ge=0, le=1)
    success_probability: float = Field(ge=0, le=1)
    model_version: str