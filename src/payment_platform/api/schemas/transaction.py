from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float = Field(gt=0)
    currency: str = "INR"
    selected_payment_method: str
    experiment_variant: str | None = None
    status: str = "initiated"


class TransactionResponse(BaseModel):
    transaction_id: str
    status: str
    experiment_variant: str