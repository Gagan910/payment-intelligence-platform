from pydantic import BaseModel, Field


class PaymentAttemptRequest(BaseModel):
    transaction_id: str
    payment_method: str
    attempt_number: int = Field(ge=1)


class PaymentAttemptResponse(BaseModel):
    attempt_id: str
    transaction_id: str
    payment_method: str
    attempt_number: int
    outcome: str