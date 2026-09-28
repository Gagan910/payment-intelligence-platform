from itertools import count

from locust import HttpUser, between, task


TRANSACTION_COUNT = 50
_transaction_counter = count()


class PaymentPredictionUser(HttpUser):
    wait_time = between(1, 2)

    def on_start(self) -> None:
        user_index = next(_transaction_counter)

        self.transaction_id = (
            f"load_test_prediction_{user_index % TRANSACTION_COUNT}"
        )

        self.prediction_payload = {
            "transaction_id": self.transaction_id,
            "amount": 500,
            "merchant_category": "electronics",
            "payment_method": "debit_card",
            "user_segment": "regular",
            "device_type": "mobile",
            "network_quality": "poor",
            "hour_of_day": 0,
            "day_of_week": 2,
            "retry_count": 0,
            "transaction_velocity": 2,
            "user_method_success_rate": 0.90,
            "merchant_method_success_rate": 0.92,
        }

    @task
    def predict_payment_failure(self):
        with self.client.post(
            "/predict",
            json=self.prediction_payload,
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"Unexpected status code: {response.status_code}"
                )