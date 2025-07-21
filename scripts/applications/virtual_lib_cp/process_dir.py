import argparse
from pathlib import Path
import shutil


def organize_xyz_files(directory: Path, molecule_id: str):
    """
    Organizes XYZ files within a specified directory.

    This function performs two main operations:
    1. Moves all .xyz files that do not contain "opt" in their name
       into a newly created 'gen' subdirectory.
    2. Renames files containing "_opt.xyz" by extracting a numerical part
       and formatting them as "co_cpx_{molecule_id}_{xxxxx}.xyz",
       where 'xxxxx' is a zero-padded 5-digit number.

    Args:
        directory (Path): The path to the directory containing the XYZ files.
        molecule_id (str): A string identifier to be included in the new filenames.
    """
    # Create the 'gen' subdirectory if it doesn't already exist.
    gen_dir = directory / "gen"
    gen_dir.mkdir(exist_ok=True)

    # Move non-optimized XYZ files to the 'gen' subdirectory.
    for file_path in directory.glob("*.xyz"):
        # Check if "opt" is NOT in the stem (filename without extension).
        if "opt" not in file_path.stem:
            # Move the file to the 'gen' directory.
            shutil.move(str(file_path), gen_dir / file_path.name)

    # Rename optimized XYZ files.
    for file_path in directory.glob("*opt.xyz"):
        # Split the filename stem by '_' to extract parts.
        parts = file_path.stem.split("_")
        # Ensure there are enough parts to extract the numerical identifier.
        if len(parts) >= 3:
            # The numerical part is expected to be the second-to-last element,
            # which is then zero-padded to 5 digits.
            numerical_part = parts[-2].zfill(5)
            # Construct the new filename.
            new_name = f"co_cpx_{molecule_id}_{numerical_part}.xyz"
            # Rename the file within the same directory.
            file_path.rename(directory / new_name)


if __name__ == "__main__":
    """
    Main execution block for the XYZ file organization script.

    Parses command-line arguments for the input directory and molecule ID,
    validates the directory path, and then calls the organization function.
    """
    parser = argparse.ArgumentParser(description="Organize XYZ files.")
    parser.add_argument(
        "directory_path",
        type=str,
        help="Path to the directory containing XYZ files."
    )
    parser.add_argument(
        "molecule_id",
        type=str,
        help="Molecule ID to be used in renaming optimized files."
    )
    args = parser.parse_args()

    # Convert the input directory string to a Path object for easier manipulation.
    target_directory = Path(args.directory_path)

    # Validate that the provided path is an existing directory.
    if not target_directory.is_dir():
        raise ValueError(f"The provided path '{target_directory}' is not a valid directory.")

    # Call the main organization function.
    organize_xyz_files(target_directory, args.molecule_id)
