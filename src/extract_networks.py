# src/extract_networks.py
# This is just used to extract all A set containing all unique asset symbols found. So we can classify as stable and unstable
             
import json
import sys

def extract_unique_asset_symbols(filepath):

    print(f"Attempting to read data from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file was not found at '{filepath}'. Please check the path.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{filepath}' is not a valid JSON file.")
        return None
    
    if not isinstance(data, list):
        print("Error: JSON data is not in the expected format (a list of objects).")
        return None

    # Use a set comprehension to efficiently find unique values.
    # It iterates through each record, accesses the nested 'actionData',
    # and gets the 'assetSymbol' if it exists.
    unique_symbols = {
        record['actionData']['assetSymbol']
        for record in data
        if 'actionData' in record and 'assetSymbol' in record['actionData']
    }
    
    return unique_symbols


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/extract_networks.py <path_to_json_file>")
        print("Example: python src/extract_networks.py data/user-wallet-transactions.json")
        sys.exit(1)

    json_filepath = sys.argv[1]
    
    # --- Extract Asset Symbols ---
    print("\n--- Extracting unique 'assetSymbol' values ---")
    asset_symbols = extract_unique_asset_symbols(json_filepath)
    
    if asset_symbols:
        print("✅ Found the following unique asset symbols:")
        print(asset_symbols)

