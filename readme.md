# Aave V2 Wallet Credit Scoring Engine

**Author:** [Your Name]
**Date:** 21-07-2025
**Version:** 1.0

## Project Overview

This project provides a one-step script to analyze raw Aave V2 transaction data and generate a credit score (from 0 to 1000) for each unique user wallet. The script uses a machine learning model to evaluate a wallet's historical behavior, assigning higher scores to reliable users and lower scores to those exhibiting risky or irresponsible patterns.

The entire pipeline—from data loading to scoring and generating an analysis graph—is contained within the single `src/scoring.py` script.

## Scoring Logic, Validation, and Transparency

This section details the logic behind the credit score, how its validity is confirmed, and how the model can be extended.

### Core Logic: Anomaly Detection

The scoring engine uses an **Isolation Forest**, a machine learning model designed for **anomaly detection**. Our core assumption is:

> **Responsible financial behavior is the norm, while high-risk behavior is an anomaly.**

Instead of defining hard-coded rules (e.g., "if liquidation, then score = 50"), the model learns the patterns of the "normal" majority of users and then identifies which wallets deviate significantly from this norm. Wallets that are flagged as anomalies receive a lower score.

### Feature Engineering for Transparency

To ensure the model's decisions are based on sound financial principles, we feed it a specific set of engineered features. This makes the logic transparent, as we know exactly what characteristics are being judged:

* **`liquidation_count`**: The number of times a wallet has been liquidated. This is the most powerful indicator of high-risk behavior.
* **`repay_to_borrow_ratio`**: A measure of financial responsibility. A ratio near 1.0 indicates a user reliably repays their debts.
* **`net_worth_proxy_usd`**: The net difference between total deposits and borrows. A positive value suggests financial stability.
* **`wallet_age_days` & `unique_days_active`**: Measures of history and consistent engagement with the protocol. Longer-term, consistent users are generally more reliable.

### Validation: How We Know the Score is Meaningful

The model's output is validated by confirming a direct correlation between scores and on-chain behavior:

1.  **Low Scores Correlate with Liquidations:** When we manually inspect the lowest-scoring wallets (e.g., scores < 200), we consistently find they have at least one `liquidation_call` event in their history. This confirms the model correctly identifies and penalizes the highest-risk users.
2.  **High Scores Correlate with Safe Behavior:** The highest-scoring wallets (e.g., scores > 800) invariably have **zero** liquidations, a long history of activity, and a healthy repayment ratio.
3.  **Logical Consistency:** The model correctly scores a new wallet with one safe deposit higher than a very active wallet that has been liquidated. This proves the logic prioritizes **safety over raw activity**, which is crucial for a credit score.

### Extensibility: How to Improve the Model

The current model is a robust baseline. It can be easily extended for greater accuracy or to evaluate different behaviors:

* **Add More Features:** The `engineer_features` function in `src/scoring.py` can be modified to include new metrics, such as:
    * **Health Factor Analysis:** Tracking how close a user's health factor gets to the liquidation threshold (1.0).
    * **Asset Diversity:** Analyzing the variety of assets a user deposits or borrows.
    * **Flash Loan Usage:** Identifying wallets that frequently use flash loans, which can be a sign of sophisticated (and potentially risky) strategies.
* **Tune the Model:** The `IsolationForest` model has parameters (like `contamination`) that can be tuned to make it more or less sensitive to anomalies.
* **Swap the Model:** The scoring function can be replaced with a different model entirely, such as a rules-based scorecard (similar to CIBIL/Experian) for maximum transparency, or a more complex model like a Gradient Boosting Tree if labeled data becomes available.

## How to Run the Project

### Prerequisites

* Python 3.8+
* A Python virtual environment

### Setup & Execution

1.  **Clone the repository and navigate into the project folder.**
2.  **Set up the environment and install dependencies:**
    ```bash
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install required libraries
    pip install -r requirements.txt
    ```
3.  **Place your data file** (e.g., `user-wallet-transactions.json`) into the `/data` directory.
4.  **Run the script:**
    ```bash
    python src/scoring.py data/your_data_file.json
    ```

The script will execute the entire pipeline and generate `wallet_scores.csv` and `score_distribution.png` in the root project directory.