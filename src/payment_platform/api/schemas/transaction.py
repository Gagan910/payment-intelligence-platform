from pydantic import BaseModel, Field


class TransactionContextRequest(BaseModel):
    device_type: str
    network_quality: str
    retry_count: int = Field(ge=0)
    transaction_velocity: int = Field(ge=0)
    user_method_success_rate: float = Field(ge=0, le=1)
    merchant_method_success_rate: float = Field(ge=0, le=1)


class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float = Field(gt=0)
    currency: str = "INR"
    selected_payment_method: str
    experiment_variant: str | None = None
    status: str = "initiated"
    context: TransactionContextRequest


class TransactionResponse(BaseModel):
    transaction_id: str
    status: str
    experiment_variant: str