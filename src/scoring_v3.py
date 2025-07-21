# DeFi Credit Scoring Engine v3
# Author: [Your Name]
# Date: 21-07-2025
# Description: This ultimate script evaluates wallet risk based on Health Factor,
#              collateral volatility, and asset diversification.

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def load_and_preprocess(filepath):
    """
    Loads and preprocesses the raw transaction data.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data from '{filepath}'...")
    
    try:
        df = pd.read_json(filepath)
    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}")
        sys.exit(1)

    action_data_df = pd.json_normalize(df['actionData'])
    df = pd.concat([df.drop(columns=['actionData']), action_data_df], axis=1)

    df['amount_numeric'] = pd.to_numeric(df['amount'], errors='coerce')
    df['price_usd_numeric'] = pd.to_numeric(df['assetPriceUSD'], errors='coerce')

    # Using a set for efficient lookup of stablecoins
    stablecoins = {'USDC', 'USDT', 'DAI'}

    def adjust_amount(row):
        symbol = row.get('assetSymbol', '')
        # Stablecoins often have 6 decimals, while most others have 18.
        if symbol in stablecoins:
            return row['amount_numeric'] * 1e-6
        return row['amount_numeric'] * 1e-18

    df['adjusted_amount'] = df.apply(adjust_amount, axis=1)
    df['usd_value'] = df['adjusted_amount'] * df['price_usd_numeric']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

def engineer_ultimate_features(df):
    """
    Engineers a comprehensive feature set including Health Factor,
    collateral risk, and asset diversification.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Engineering ultimate features for {df['userWallet'].nunique()} wallets...")
    
    wallets = {}
    # Define the set of known stablecoins for risk analysis
    stablecoins = {'USDC', 'USDT', 'DAI'}
    
    for wallet_address, group in df.groupby('userWallet'):
        group = group.sort_values('timestamp')
        
        # Initialize variables to track state over time for each wallet
        current_deposits_usd = 0
        current_borrows_usd = 0
        min_health_factor = 1000  # Start with a high, safe number
        
        total_volatile_deposit_usd = 0
        total_deposit_usd = 0

        # Simulate the wallet's history transaction by transaction
        for _, tx in group.iterrows():
            action = tx['action']
            value = tx['usd_value']
            symbol = tx.get('assetSymbol', '')

            if action == 'deposit':
                current_deposits_usd += value
                total_deposit_usd += value
                # Check if the deposited asset is volatile
                if symbol not in stablecoins:
                    total_volatile_deposit_usd += value
            elif action == 'redeemunderlying':
                current_deposits_usd -= value
            elif action == 'borrow':
                current_borrows_usd += value
            elif action == 'repay':
                current_borrows_usd -= value
            elif action == 'liquidationcall':
                current_borrows_usd -= value
                current_deposits_usd -= value * 1.05 # Assume a 5% liquidation penalty

            current_deposits_usd = max(0, current_deposits_usd)
            current_borrows_usd = max(0, current_borrows_usd)

            # Calculate Health Factor after each transaction if there's an active loan
            if current_borrows_usd > 0:
                # Health Factor = (Collateral * Liquidation Threshold) / Borrows
                # We use 0.8 as an average liquidation threshold for simplicity.
                health_factor = (current_deposits_usd * 0.8) / current_borrows_usd
                if health_factor < min_health_factor:
                    min_health_factor = health_factor

        # --- Finalize features for the wallet ---
        actions = group['action'].value_counts()
        liquidation_count = actions.get('liquidationcall', 0)
        
        # If a user never borrowed, their health is perfect. Cap at 10 for the model.
        if min_health_factor == 1000:
            min_health_factor = 10 

        # NEW FEATURE: Calculate the ratio of volatile assets in their total deposits
        volatile_collateral_ratio = 0
        if total_deposit_usd > 0:
            volatile_collateral_ratio = total_volatile_deposit_usd / total_deposit_usd
            
        # NEW FEATURE: Count how many different assets they've used
        asset_diversification = group['poolId'].nunique()

        wallets[wallet_address] = {
            'wallet_age_days': (group['timestamp'].max() - group['timestamp'].min()).days + 1,
            'total_tx_count': len(group),
            'liquidation_count': liquidation_count,
            'minimum_health_factor': min_health_factor,
            'volatile_collateral_ratio': volatile_collateral_ratio,
            'asset_diversification': asset_diversification
        }
        
    features_df = pd.DataFrame.from_dict(wallets, orient='index')
    features_df.replace([np.inf, -np.inf], 10, inplace=True) # Clean up potential infinities
    return features_df

def calculate_scores_v3(features_df):
    """
    Calculates scores using the ultimate feature set.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training ultimate model and calculating scores...")
    
    # Use all our new features in the model
    model_features = [
        'wallet_age_days', 'total_tx_count',
        'liquidation_count', 'minimum_health_factor',
        'volatile_collateral_ratio', 'asset_diversification'
    ]
    
    X = features_df[model_features].values
    
    if X.shape[0] < 2:
        features_df['credit_score'] = 500
        return features_df

    iso_forest = IsolationForest(contamination='auto', random_state=42)
    iso_forest.fit(X)
    
    raw_scores = iso_forest.decision_function(X)
    
    scaler = MinMaxScaler(feature_range=(0, 1000))
    credit_scores = scaler.fit_transform(raw_scores.reshape(-1, 1))
    
    features_df['credit_score'] = credit_scores.astype(int)
    return features_df

def generate_score_distribution_graph(scores_df):
    """
    Generates and saves a histogram of the credit scores.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating score distribution graph...")
    plt.figure(figsize=(12, 7))
    plt.hist(scores_df['credit_score'], bins=10, range=(0, 1000), edgecolor='black')
    plt.title('Distribution of Wallet Credit Scores (v3 Model)', fontsize=16)
    plt.xlabel('Credit Score Bins', fontsize=12)
    plt.ylabel('Number of Wallets', fontsize=12)
    plt.xticks(range(0, 1001, 100))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('score_distribution_v3.png')
    print("✅ Graph saved successfully to 'score_distribution_v3.png'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/scoring_v3.py <path_to_json_file>")
        sys.exit(1)
        
    input_filepath = sys.argv[1]
    
    processed_data = load_and_preprocess(input_filepath)
    wallet_features = engineer_ultimate_features(processed_data)
    final_scores_df = calculate_scores_v3(wallet_features)
    
    output_filepath = 'wallet_scores_v3.csv'
    final_scores_df.index.name = 'userWallet'
    final_scores_df.to_csv(output_filepath)
    
    print(f"\n✅ Ultimate scores saved successfully to '{output_filepath}'")
    print("\n--- Sample Ultimate Scores ---")
    print(final_scores_df[['credit_score', 'minimum_health_factor', 'volatile_collateral_ratio', 'asset_diversification']].head())
    print("-" * 70)

    generate_score_distribution_graph(final_scores_df)
