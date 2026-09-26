from fastapi import FastAPI

from payment_platform.api.routes.prediction import router as prediction_router
from payment_platform.api.routes.recommendation import (
    router as recommendation_router,
)
from payment_platform.api.routes.payment_attempt import (
    router as payment_attempt_router,
)
from payment_platform.api.routes.transaction import router as transaction_router


app = FastAPI(
    title="Payment Intelligence & Smart Routing Platform",
    version="0.1.0",
    description=(
        "ML-powered payment failure prediction and smart routing "
        "platform using synthetic payment data."
    ),
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


app.include_router(transaction_router)
app.include_router(prediction_router)
app.include_router(recommendation_router)
app.include_router(payment_attempt_router)