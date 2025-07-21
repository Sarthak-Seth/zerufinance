# DeFi Credit Scoring Engine
# Author: [Your Name]
# Date: 21-07-2025
# Description: This single script processes Aave V2 transaction data, generates a credit score
#              for each wallet, saves the scores to a CSV, and creates a distribution graph.

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys
import warnings
from datetime import datetime

# It's good practice to ignore harmless warnings for a cleaner console output.
warnings.filterwarnings('ignore')

def load_and_preprocess(filepath):
    """
    Loads raw transaction data from a JSON file and prepares it for feature engineering.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data from '{filepath}'...")
    
    try:
        df = pd.read_json(filepath)
    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)

    action_data_df = pd.json_normalize(df['actionData'])
    df = pd.concat([df.drop(columns=['actionData']), action_data_df], axis=1)

    df['amount_numeric'] = pd.to_numeric(df['amount'], errors='coerce')
    df['price_usd_numeric'] = pd.to_numeric(df['assetPriceUSD'], errors='coerce')

    def adjust_amount(row):
        symbol = row.get('assetSymbol', '')
        if symbol in ['USDC', 'USDT']:
            return row['amount_numeric'] * 1e-6
        return row['amount_numeric'] * 1e-18

    df['adjusted_amount'] = df.apply(adjust_amount, axis=1)
    df['usd_value'] = df['adjusted_amount'] * df['price_usd_numeric']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

def engineer_features(df):
    """
    Transforms raw transaction data into meaningful features for each wallet.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Engineering features for {df['userWallet'].nunique()} wallets...")
    
    wallets = {}
    
    for wallet, group in df.groupby('userWallet'):
        group = group.sort_values('timestamp')
        
        first_tx = group['timestamp'].min()
        last_tx = group['timestamp'].max()
        wallet_age_days = (last_tx - first_tx).days + 1
        
        actions = group['action'].value_counts()
        
        total_deposit_usd = group[group['action'] == 'deposit']['usd_value'].sum()
        total_borrow_usd = group[group['action'] == 'borrow']['usd_value'].sum()
        total_repay_usd = group[group['action'] == 'repay']['usd_value'].sum()
        
        liquidation_count = actions.get('liquidationcall', 0)
        repay_to_borrow_ratio = total_repay_usd / total_borrow_usd if total_borrow_usd > 0 else 1.0
        
        wallets[wallet] = {
            'wallet_age_days': wallet_age_days,
            'total_tx_count': len(group),
            'unique_days_active': group['timestamp'].dt.date.nunique(),
            'total_deposit_usd': total_deposit_usd,
            'total_borrow_usd': total_borrow_usd,
            'net_worth_proxy_usd': total_deposit_usd - total_borrow_usd,
            'liquidation_count': liquidation_count,
            'repay_to_borrow_ratio': repay_to_borrow_ratio
        }
        
    features_df = pd.DataFrame.from_dict(wallets, orient='index')
    return features_df

def calculate_scores(features_df):
    """
    Uses an Isolation Forest model to score wallets based on their features.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training model and calculating scores...")
    
    model_features = [
        'wallet_age_days', 'total_tx_count', 'unique_days_active',
        'net_worth_proxy_usd', 'liquidation_count', 'repay_to_borrow_ratio'
    ]
    
    X = features_df[model_features].values
    
    if X.shape[0] < 2:
        print("Warning: Not enough data to build a model. Assigning a default score.")
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

    Args:
        scores_df (pandas.DataFrame): The DataFrame containing the 'credit_score' column.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating score distribution graph...")
    
    plt.figure(figsize=(12, 7))
    plt.hist(scores_df['credit_score'], bins=10, range=(0, 1000), edgecolor='black')
    
    plt.title('Distribution of Wallet Credit Scores', fontsize=16)
    plt.xlabel('Credit Score Bins', fontsize=12)
    plt.ylabel('Number of Wallets', fontsize=12)
    plt.xticks(range(0, 1001, 100))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    graph_filepath = 'score_distribution.png'
    plt.savefig(graph_filepath)
    
    print(f"✅ Graph saved successfully to '{graph_filepath}'")

# This is the main entry point of the script.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/scoring.py <path_to_json_file>")
        sys.exit(1)
        
    input_filepath = sys.argv[1]
    
    # --- Execute the Full Pipeline ---
    processed_data = load_and_preprocess(input_filepath)
    wallet_features = engineer_features(processed_data)
    final_scores_df = calculate_scores(wallet_features)
    
    # --- Save the CSV Output ---
    output_filepath = 'wallet_scores.csv'
    final_scores_df.index.name = 'userWallet'
    final_scores_df[['credit_score']].to_csv(output_filepath)
    
    print(f"\n✅ Scores saved successfully to '{output_filepath}'")
    print("\n--- Sample Scores ---")
    print(final_scores_df[['credit_score']].head())
    print("-" * 25)

    # --- Generate the Graph Output ---
    generate_score_distribution_graph(final_scores_df)

