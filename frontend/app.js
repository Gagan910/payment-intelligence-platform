const API_BASE_URL = "http://127.0.0.1:8001";

const form = document.getElementById("checkout-form");
const amountInput = document.getElementById("amount");
const merchantSelect = document.getElementById("merchant");

const merchantName = document.getElementById("merchant-name");
const amountDisplay = document.getElementById("amount-display");

const loadingState = document.getElementById("loading-state");
const predictionPanel = document.getElementById("prediction-panel");
const recommendationPanel = document.getElementById(
    "recommendation-panel"
);
const paymentPanel = document.getElementById("payment-panel");
const resultPanel = document.getElementById("result-panel");
const errorPanel = document.getElementById("error-panel");

const successProbability = document.getElementById(
    "success-probability"
);
const failureProbability = document.getElementById(
    "failure-probability"
);

const recommendationTitle = document.getElementById(
    "recommendation-title"
);
const recommendationReason = document.getElementById(
    "recommendation-reason"
);

const currentMethodElement = document.getElementById(
    "current-method"
);
const recommendedMethodElement = document.getElementById(
    "recommended-method"
);
const expectedImprovementElement = document.getElementById(
    "expected-improvement"
);

const acceptRecommendationButton = document.getElementById(
    "accept-recommendation"
);
const keepCurrentButton = document.getElementById(
    "keep-current"
);

const selectedMethodElement = document.getElementById(
    "selected-method"
);

const payButton = document.getElementById("pay-button");

const resultTitle = document.getElementById("result-title");
const resultMessage = document.getElementById("result-message");

const newPaymentButton = document.getElementById("new-payment");

const errorMessage = document.getElementById("error-message");

let currentTransactionId = null;
let currentRecommendationId = null;
let currentPaymentMethod = null;
let currentAttemptNumber = 1;


function generateTransactionId() {
    const timestamp = Date.now();
    const randomPart = Math.random()
        .toString(36)
        .substring(2, 10);

    return `web_${timestamp}_${randomPart}`;
}


function generateUserId() {
    return `web_user_${Math.random()
        .toString(36)
        .substring(2, 10)}`;
}


function generateMerchantId() {
    return merchantSelect.value;
}


function getSelectedPaymentMethod() {
    const selected = document.querySelector(
        'input[name="payment_method"]:checked'
    );

    return selected ? selected.value : null;
}


function formatPaymentMethod(method) {
    if (!method) {
        return "--";
    }

    const names = {
        upi: "UPI",
        credit_card: "Credit Card",
        debit_card: "Debit Card",
        net_banking: "Net Banking",
        wallet: "Wallet",
    };

    return names[method] || method;
}


function formatPercentage(value) {
    if (value === null || value === undefined) {
        return "--";
    }

    return `${(value * 100).toFixed(2)}%`;
}


function formatImprovement(value) {
    if (value === null || value === undefined) {
        return "--";
    }

    return `${(value * 100).toFixed(2)} percentage points`;
}


function show(element) {
    element.classList.remove("hidden");
}


function hide(element) {
    element.classList.add("hidden");
}


function showError(message) {
    errorMessage.textContent = message;
    show(errorPanel);
}


function clearError() {
    errorMessage.textContent = "";
    hide(errorPanel);
}


function resetPanels() {
    hide(loadingState);
    hide(predictionPanel);
    hide(recommendationPanel);
    hide(paymentPanel);
    hide(resultPanel);

    clearError();
}


function setLoading(isLoading) {
    if (isLoading) {
        show(loadingState);
    } else {
        hide(loadingState);
    }

    form.querySelectorAll("input, select, button").forEach(
        (element) => {
            element.disabled = isLoading;
        }
    );
}


function updateCheckoutSummary() {
    const amount = Number(amountInput.value);

    merchantName.textContent =
        merchantSelect.options[
            merchantSelect.selectedIndex
        ].text;

    if (Number.isFinite(amount) && amount > 0) {
        amountDisplay.textContent =
            new Intl.NumberFormat("en-IN", {
                style: "currency",
                currency: "INR",
            }).format(amount);
    } else {
        amountDisplay.textContent = "₹0.00";
    }
}


async function createTransaction() {
    const amount = Number(amountInput.value);
    const paymentMethod = getSelectedPaymentMethod();

    if (!Number.isFinite(amount) || amount <= 0) {
        throw new Error("Please enter a valid payment amount.");
    }

    if (!paymentMethod) {
        throw new Error("Please select a payment method.");
    }

    const transactionId = generateTransactionId();

    const requestBody = {
        transaction_id: transactionId,
        user_id: generateUserId(),
        merchant_id: generateMerchantId(),
        amount: amount,
        currency: "INR",
        selected_payment_method: paymentMethod,
        status: "initiated",
        context: {
            device_type: "desktop",
            network_quality: "good",
            retry_count: 0,
            transaction_velocity: 2,
            user_method_success_rate: 0.90,
            merchant_method_success_rate: 0.92,
        },
    };

    const response = await fetch(
        `${API_BASE_URL}/transactions`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestBody),
        }
    );

    if (!response.ok) {
        const detail = await response.text();
        throw new Error(
            `Transaction creation failed: ${detail}`
        );
    }

    const data = await response.json();

    currentTransactionId = data.transaction_id;
    currentPaymentMethod = paymentMethod;

    return data;
}


async function requestPrediction() {
    const response = await fetch(
        `${API_BASE_URL}/predict`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                transaction_id: currentTransactionId,
            }),
        }
    );

    if (!response.ok) {
        const detail = await response.text();
        throw new Error(
            `Prediction failed: ${detail}`
        );
    }

    return await response.json();
}


async function requestRecommendation() {
    const response = await fetch(
        `${API_BASE_URL}/recommend`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                transaction_id: currentTransactionId,
            }),
        }
    );

    if (!response.ok) {
        const detail = await response.text();
        throw new Error(
            `Recommendation failed: ${detail}`
        );
    }

    return await response.json();
}


function renderPrediction(prediction) {
    successProbability.textContent =
        formatPercentage(prediction.success_probability);

    failureProbability.textContent =
        formatPercentage(prediction.failure_probability);

    show(predictionPanel);
}


function renderRecommendation(recommendation) {
    currentRecommendationId =
        recommendation.recommendation_id;

    currentPaymentMethod =
        recommendation.current_method;

    currentMethodElement.textContent =
        formatPaymentMethod(
            recommendation.current_method
        );

    recommendedMethodElement.textContent =
        formatPaymentMethod(
            recommendation.recommended_method
        );

    expectedImprovementElement.textContent =
        formatImprovement(
            recommendation.expected_improvement
        );

    recommendationReason.textContent =
        recommendation.reason || "";

    if (
        recommendation.recommendation_action ===
        "RECOMMEND_ALTERNATIVE"
    ) {
        recommendationTitle.textContent =
            `We recommend ${formatPaymentMethod(
                recommendation.recommended_method
            )}`;

        show(acceptRecommendationButton);
        show(keepCurrentButton);
    } else {
        recommendationTitle.textContent =
            "Keep your current payment method";

        hide(acceptRecommendationButton);
        show(keepCurrentButton);
    }

    show(recommendationPanel);
}


async function recordRecommendationDecision(accepted) {
    if (!currentRecommendationId) {
        throw new Error(
            "No recommendation is available."
        );
    }

    const response = await fetch(
        `${API_BASE_URL}/recommend/decision`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                recommendation_id:
                    currentRecommendationId,
                accepted: accepted,
            }),
        }
    );

    if (!response.ok) {
        const detail = await response.text();
        throw new Error(
            `Recommendation decision failed: ${detail}`
        );
    }

    const data = await response.json();

    currentPaymentMethod =
        data.selected_payment_method;

    // Keep the payment-method radio button synchronized
    // with the method actually selected for payment.
    const selectedRadio = document.querySelector(
        `input[name="payment_method"][value="${currentPaymentMethod}"]`
    );

    if (selectedRadio) {
        selectedRadio.checked = true;
    }

    selectedMethodElement.textContent =
        formatPaymentMethod(
            currentPaymentMethod
        );

    hide(recommendationPanel);
    show(paymentPanel);
}


async function simulatePayment() {
    if (!currentTransactionId) {
        throw new Error(
            "No active transaction exists."
        );
    }

    payButton.disabled = true;
    payButton.textContent = "Processing...";

    const response = await fetch(
        `${API_BASE_URL}/payment-attempts`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                transaction_id:
                    currentTransactionId,
                payment_method:
                    currentPaymentMethod,
                attempt_number:
                    currentAttemptNumber,
            }),
        }
    );

    if (!response.ok) {
        const detail = await response.text();

        payButton.disabled = false;
        payButton.textContent =
            "Simulate Payment";

        throw new Error(
            `Payment attempt failed: ${detail}`
        );
    }

    const data = await response.json();

    renderPaymentResult(data);
}


function renderPaymentResult(payment) {
    hide(paymentPanel);

    if (payment.outcome === "success") {
        resultTitle.textContent =
            "Payment Successful";

        resultMessage.textContent =
            `Simulated payment completed using ${formatPaymentMethod(
                payment.payment_method
            )}.`;
    } else {
        resultTitle.textContent =
            "Payment Failed";

        resultMessage.textContent =
            `The simulated payment failed using ${formatPaymentMethod(
                payment.payment_method
            )}.`;
    }

    show(resultPanel);

    payButton.disabled = false;
    payButton.textContent = "Simulate Payment";
}


async function handleCheckout(event) {
    event.preventDefault();

    resetPanels();
    setLoading(true);

    try {
        await createTransaction();

        const prediction =
            await requestPrediction();

        renderPrediction(prediction);

        const recommendation =
            await requestRecommendation();

        renderRecommendation(recommendation);
    } catch (error) {
        console.error(error);
        showError(
            error.message ||
            "Something went wrong."
        );
    } finally {
        setLoading(false);
    }
}


function resetCheckout() {
    form.reset();

    currentTransactionId = null;
    currentRecommendationId = null;
    currentPaymentMethod = "upi";
    currentAttemptNumber = 1;

    resetPanels();

    updateCheckoutSummary();
}


amountInput.addEventListener(
    "input",
    updateCheckoutSummary
);

merchantSelect.addEventListener(
    "change",
    updateCheckoutSummary
);

form.addEventListener(
    "submit",
    handleCheckout
);

acceptRecommendationButton.addEventListener(
    "click",
    async () => {
        try {
            await recordRecommendationDecision(true);
        } catch (error) {
            console.error(error);
            showError(error.message);
        }
    }
);

keepCurrentButton.addEventListener(
    "click",
    async () => {
        try {
            await recordRecommendationDecision(false);
        } catch (error) {
            console.error(error);
            showError(error.message);
        }
    }
);

payButton.addEventListener(
    "click",
    async () => {
        try {
            await simulatePayment();
        } catch (error) {
            console.error(error);
            showError(error.message);
        }
    }
);

newPaymentButton.addEventListener(
    "click",
    resetCheckout
);

updateCheckoutSummary();