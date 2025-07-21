import os
import argparse
import subprocess
import shutil

def align_and_concatenate(
    reference_file: str,
    template_file: str,
    ligand_dir: str,
    output_dir: str,
    it_indices: str,
    rt_indices: str
) -> None:
    """
    Aligns a set of ligand XYZ files to a reference structure using rmsdAlign.py,
    and then concatenates the aligned ligand coordinates with a template XYZ file.

    The process involves:
    1. Creating the output directory if it doesn't exist.
    2. Iterating through each XYZ file in the ligand directory.
    3. Checking if the ligand file is empty.
    4. Running `rmsdAlign.py` to align the current ligand to the reference file
       based on specified atom indices. The aligned coordinates are saved to a
       temporary file.
    5. Reading the atom counts and coordinates from the template file and the
       temporarily aligned ligand file.
    6. Combining the coordinates and updating the total atom count.
    7. Writing the combined structure to a new XYZ file in the output directory.
    8. Cleaning up the temporary aligned file.

    Args:
        reference_file (str): Path to the reference .xyz file for alignment.
        template_file (str): Path to the template .xyz file whose coordinates
                             will be prepended to the aligned ligand.
        ligand_dir (str): Path to the directory containing the ligand .xyz files
                          to be aligned and concatenated.
        output_dir (str): Path to the directory where the combined .xyz files
                          will be saved.
        it_indices (str): Comma-separated string of 1-based atom indices in the
                          target (ligand) file to be used for RMSD alignment.
        rt_indices (str): Comma-separated string of 1-based atom indices in the
                          reference file to be used for RMSD alignment.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ligand_files = [f for f in os.listdir(ligand_dir) if f.endswith('.xyz')]

    for ligand_file in ligand_files:
        ligand_path = os.path.join(ligand_dir, ligand_file)
        tmp_align_path = "tmp_aligned_ligand.xyz"

        with open(ligand_path, 'r') as ligand:
            ligand_lines = ligand.readlines()
            if not ligand_lines:
                print(f"Warning: {ligand_file} is empty. Skipping.")
                continue
            try:
                # Check if the ligand file is empty or malformed
                num_atoms_ligand = int(ligand_lines[0].strip())
                if num_atoms_ligand == 0:
                    print(f"Warning: {ligand_file} declares 0 atoms. Skipping.")
                    continue
            except ValueError:
                print(f"Error: Could not parse atom count from {ligand_file}. Skipping.")
                continue

        # Construct the rmsdAlign.py command
        command = [
            "rmsdAlign.py", ligand_path, 
            "-r", reference_file, 
            "-it", it_indices,
            "-rt", rt_indices,
            "-n", 
            "-o", tmp_align_path
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Aligned {ligand_file} to {reference_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error aligning {ligand_file}: {e}. Command: {' '.join(command)}")
            if os.path.exists(tmp_align_path):
                os.remove(tmp_align_path)
            continue
        except FileNotFoundError:
            print("Error: rmsdAlign.py not found. Please ensure it's in your PATH.")
            return 

        # Combine coordinates from template_file and tmp_align.xyz
        output_file = os.path.join(output_dir, ligand_file)
        with open(template_file, 'r') as template, open(tmp_align_path, 'r') as tmp_align:
            template_lines = template.readlines()
            tmp_align_lines = tmp_align.readlines()
            
            template_atom_count = int(template_lines[0].strip()) if template_lines else 0
            align_atom_count = int(tmp_align_lines[0].strip()) if tmp_align_lines else 0

            combined_coordinates = template_lines[2:] + tmp_align_lines[2:]

            with open(output_file, 'w') as outfile:
                total_atom_count = template_atom_count + align_atom_count
                outfile.write(f"{total_atom_count}\n") # Write updated atom count
                outfile.write(template_lines[1])  # Write comment line
                outfile.writelines(combined_coordinates)  # Write combined coordinates

        print(f"Saved concatenated file to {output_file}")
        
        if os.path.exists(tmp_align_path):
            os.remove(tmp_align_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align ligand structures and concatenate with a template.")
    parser.add_argument(
        "reference", 
        type=str, 
        help="Path to the reference .xyz file"
    )
    parser.add_argument(
        "template", 
        type=str, 
        help="Path to the template .xyz file"
    )
    parser.add_argument(
        "ligand_dir", 
        type=str, 
        help="Path to the directory containing ligand .xyz files"
    )
    parser.add_argument(
        "output_dir", 
        type=str, 
        help="Path to the output directory"
    )
    parser.add_argument(
        "--it_indices", 
        type=str, 
        default="26,27,1,2,3,4,5,6,7,16,17", 
        help="Indices for the target file (default: 26,27,1,2,3,4,5,6,7,16,17)"
    )
    parser.add_argument(
        "--rt_indices", 
        type=str, 
        default="26,27,1,2,3,4,5,6,7,16,17", 
        help="Indices for the reference file (default: 26,27,1,2,3,4,5,6,7,16,17)"
    )
    args = parser.parse_args()

    align_and_concatenate(args.reference, args.template, args.ligand_dir, args.output_dir, args.it_indices, args.rt_indices)
