from fastapi import FastAPI


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
    """Return API health status."""
    return {"status": "healthy"}