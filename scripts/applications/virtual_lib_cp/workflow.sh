#!/bin/bash

# This script orchestrates a molecular simulation workflow.
# It takes 'name' and 'working_path' as command-line arguments.

# Parse command-line arguments
name="$1"
working_path="$2"

# Check if arguments are provided
if [ -z "$name" ] || [ -z "$working_path" ]; then
  echo "Usage: $0 <name> <working_path>"
  echo "Example: $0 124535_alya_0.4 generated/raw/124535_alya_0.4"
  exit 1
fi

# Step 1: Filter molecules
python filter_mol.py -i "${working_path}" --c control_configs/connector.yaml 

# Step 2: Dock ligand
python dock_ligand.py ts_template/cp_co_cpx.xyz ts_template/co.xyz "${working_path}" "${working_path}/${name}_co_cpx"

# Step 3: Send files to remote server
python ../../auxillary/calc/send_remote.py -r /home/worakul/RF/applications/cp -l "${working_path}/${name}_co_cpx"



# # Step 4: Perform xTB optimization (assuming it runs on remote or locally after transfer)
# python xtb_opt.py "${name}_co_cpx"

# # Step 5: Process optimized XYZ files
# python process_xtbopt.xyz "${name}_co_cpx" "${name}"

# # Step 6: Filter co_cpx molecules
# python filter_mol_co_cpx.py --c connector_co_cpx.yaml -i "${name}_co_cpx"

# # Step 7: Get steric descriptors
# python get_steric_desc.py --i "${name}_co_cpx" --o "${name}.csv"

# echo "Workflow completed successfully for molecule: ${name} in path: ${working_path}"
