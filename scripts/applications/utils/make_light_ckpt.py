import torch
import argparse
import pickle

def make_light_checkpoint(input_file, output_file):
    """
    Loads a checkpoint file, keeps only 'hyperparameters' and 'ema_model',
    and saves it to a new, lighter checkpoint file.
    """
    print(f"Loading checkpoint from {input_file}...")
    try:
        # map_location='cpu' prevents loading onto a GPU by default
        data = torch.load(input_file, map_location='cpu')
    except Exception as e:
        print(f"Error loading with torch.load: {e}")
        print("Attempting to load with pickle...")
        try:
            with open(input_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as pickle_e:
            print(f"Error loading with pickle: {pickle_e}")
            return

    print(f"Original dictionary keys: {list(data.keys())}")

    # The user mentioned both 'hyperparameters' and 'hyperparamters' (with a typo)
    hyperparameters_key = None
    if 'hyperparameters' in data:
        hyperparameters_key = 'hyperparameters'
    elif 'hyperparamters' in data:
        hyperparameters_key = 'hyperparamters'
    else:
        print("Warning: Neither 'hyperparameters' nor 'hyperparamters' key found.")

    if 'ema_model' not in data:
        print("Warning: 'ema_model' key not found.")

    # Create the new dictionary, ensuring keys exist before access
    light_data = {}
    if hyperparameters_key:
        # Consistently use the correct spelling 'hyperparameters' in the new file
        light_data['hyperparameters'] = data[hyperparameters_key]
    if 'ema_model' in data:
        light_data['ema_model'] = data['ema_model']

    # Remove dataset references from hyperparameters to make the file lighter
    if 'hyperparameters' in light_data and light_data['hyperparameters'] is not None:
        hp_dict = light_data['hyperparameters']
        keys_to_remove = ['train_set', 'valid_set', 'test_set']
        for key in keys_to_remove:
            if key in hp_dict:
                del hp_dict[key]
                print(f"Removed '{key}' from hyperparameters.")

    if not light_data:
        print("Could not find any of the specified keys. Aborting.")
        return
        
    print(f"New dictionary keys will be: {list(light_data.keys())}")

    print(f"Saving light checkpoint to {output_file}...")
    torch.save(light_data, output_file)
    print("Light checkpoint saved successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Make a PyTorch checkpoint file lighter by keeping only 'ema_model' and 'hyperparameters'."
    )
    parser.add_argument(
        '--input', type=str, required=True, help='Path to the input checkpoint file.'
    )
    parser.add_argument(
        '--output', type=str, required=True, help='Path for the new lighter checkpoint file.'
    )
    args = parser.parse_args()

    make_light_checkpoint(args.input, args.output)
