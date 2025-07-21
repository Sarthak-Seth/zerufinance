# python src/generate_graph.py
import pandas as pd
import matplotlib.pyplot as plt

# Make sure to use the correct project folder name if you changed it
df = pd.read_csv('wallet_scores.csv')
plt.figure(figsize=(12, 7))

plt.hist(df['credit_score'], bins=10, range=(0, 1000), edgecolor='black')

plt.title('Distribution of Wallet Credit Scores', fontsize=16)
plt.xlabel('Credit Score Bins', fontsize=12)
plt.ylabel('Number of Wallets', fontsize=12)
plt.xticks(range(0, 1001, 100))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the graph as an image in the main project folder
plt.savefig('score_distribution.png')

print("✅ Graph saved as score_distribution.png")