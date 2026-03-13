import os
from pathlib import Path 
# List of PDB files to rename
cur_dir = Path('/cta/users/guludogan/ProteinWorkshop/proteinworkshop/data/GeneOntology/pdb')
# Loop through each file and rename it
for file in cur_dir.iterdir():
    # Split the filename to extract the new name format
    new_name = "_".join(file.name.split("_")[:-1]) + ".pdb"
    try:
        os.rename(str(file), str(cur_dir / new_name))
        print(f"Renamed: {file} -> {new_name}")
    except FileNotFoundError:
        print(f"File not found: {file}")
    except Exception as e:
        print(f"Error renaming {file}: {e}")