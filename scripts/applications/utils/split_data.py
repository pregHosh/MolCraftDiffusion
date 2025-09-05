
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os

def random_split(df, test_size):
    """
    Performs a random split of the dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        tuple: A tuple containing the training and testing DataFrames.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df

def main():
    parser = argparse.ArgumentParser(description='Split a CSV file into training and testing sets.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--split_strategy', type=str, default='random', choices=['random'], help='The splitting strategy to use.')
    parser.add_argument('--test_size', type=float, default=0.2, help='The proportion of the dataset to include in the test split.')
    args = parser.parse_args()

    # Read the input CSV file
    df = pd.read_csv(args.input_csv)

    # Split the data
    if args.split_strategy == 'random':
        train_df, test_df = random_split(df, args.test_size)
    else:
        raise ValueError(f'Unknown split strategy: {args.split_strategy}')

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the output CSV files
    train_df.to_csv(f'{args.output_dir}/train.csv', index=False)
    test_df.to_csv(f'{args.output_dir}/test.csv', index=False)

if __name__ == '__main__':
    main()
