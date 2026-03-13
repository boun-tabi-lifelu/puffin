import requests 
import json
import pandas as pd
def download_annotations(pdb_id):
    url = f"https://www.ebi.ac.uk/interpro/api/entry/InterPro/structure/PDB/{pdb_id.lower()}/"
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        try:
            # Parse the JSON response and return it
            return response.json()
        except ValueError:
            # Handle the case where the response is not valid JSON
            print(f"Error: Could not decode JSON for PDB ID {pdb_id}")
            return None
    else:
        # Handle unsuccessful request
        print(f"Error: Failed to retrieve annotations for PDB ID {pdb_id}. Status code: {response.status_code}")
        return None

def download_and_store_annotations(pdb_ids, output_file):
    all_annotations = {}

    for pdb_id in pdb_ids:
        print(f"Downloading annotations for PDB ID: {pdb_id}")
        annotations = download_annotations(pdb_id)

        if annotations:
            all_annotations[pdb_id] = annotations

    # Save the combined annotations to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(all_annotations, json_file, indent=4)

    print(f"Annotations for {len(pdb_ids)} PDB IDs have been saved to {output_file}")


df = pd.read_csv('data/GeneOntology/nrPDB-GO_test.txt', header=None)
df[0] = df[0].apply(lambda s: s.split('-')[0])
download_and_store_annotations(df[0].tolist(), 'data/GeneOntology/test_interpro.json')

