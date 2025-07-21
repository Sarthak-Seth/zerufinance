Aave V2 Wallet Credit Scoring Engine
Author: [Your Name]
Date: 21-07-2025
Version: 3.0

Project Overview
This project provides a suite of scripts to analyze raw Aave V2 transaction data and generate a credit score (from 0 to 1000) for each unique user wallet. The project showcases an evolution of three machine learning models, each adding a new layer of sophistication to how a wallet's on-chain behavior is evaluated.

The core of the project is to assign higher scores to reliable users and lower scores to those exhibiting risky or irresponsible patterns, using the Isolation Forest anomaly detection algorithm as the foundation.

Model Evolution & Scoring Logic
The project evolved through three distinct models, each building upon the last to create a more nuanced and accurate risk profile.

Model v1: The Baseline (scoring.py)
The first model establishes a baseline by focusing on the most direct indicators of risk and reliability.

Core Logic: Identifies wallets with major negative events (liquidations) as anomalies.

Key Features:

liquidation_count: The number of times a wallet was liquidated.

repay_to_borrow_ratio: A measure of debt repayment reliability.

net_worth_proxy_usd: The net difference between total deposits and borrows.

wallet_age_days: The age of the wallet's on-chain history.

Model v2: Advanced Risk Analysis (scoring_v2.py)
The second model introduces a more proactive measure of risk by analyzing how close a user gets to financial distress.

Core Logic: Moves beyond just counting liquidations to measure a user's ongoing risk management.

Key New Feature:

minimum_health_factor: The script simulates the wallet's entire history to find the lowest point its Health Factor ever reached. A user who constantly hovers near the liquidation threshold (1.0) is identified as riskier than one who maintains a high, safe Health Factor.

Model v3: Ultimate Behavioral Profile (scoring_v3.py)
The final and most robust model incorporates the context of what assets a user interacts with, adding another layer of behavioral analysis.

Core Logic: Understands that how you use assets is as important as how much you borrow.

Key New Features:

volatile_collateral_ratio: Calculates the percentage of a user's collateral that is in volatile assets (like WETH, WMATIC) versus stablecoins (like USDC, DAI). A higher ratio indicates a riskier collateral profile.

asset_diversification: Counts the number of unique assets a user has interacted with. A higher number can indicate a more sophisticated and engaged user.

Validation and Transparency
The validity of all three models is confirmed by observing a direct correlation between scores and on-chain behavior.

Low Scores consistently correlate with high-risk indicators like liquidations or a very low minimum Health Factor.

High Scores consistently correlate with safe behaviors like zero liquidations, a high Health Factor, and a long, active history.

Logical Progression: The scores from v1 to v3 become progressively more nuanced, correctly identifying users who are subtly risky even if they have never been liquidated.

How to Run the Project
Prerequisites
Python 3.8+

A Python virtual environment

Setup & Execution
Clone the repository and navigate into the project folder.

Set up the environment and install dependencies:

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required libraries
pip install -r requirements.txt

Place your data file (e.g., user-wallet-transactions.json) into the /data directory.

Run the desired scoring script:

To run the v1 model:

python src/scoring.py data/your_data_file.json

To run the v2 model (with Health Factor):

python src/scoring_v2.py data/your_data_file.json

To run the v3 model (Ultimate):

python src/scoring_v3.py data/your_data_file.json

Each script will generate its own corresponding wallet_scores_vX.csv and score_distribution_vX.png files.