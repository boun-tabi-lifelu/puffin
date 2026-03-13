import pymol2
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import seaborn as sns
import networkx as nx
from torch_geometric.utils import to_networkx
from typing import List 
from itertools import groupby
from PIL import Image
from collections import Counter
from tqdm import tqdm

sns.set()


# Define the cluster to character mapping
def cluster_to_char(cluster):
    """
    Convert a cluster ID to a representative character.
    
    Parameters:
        cluster (int): Cluster ID.

    Returns:
        str: Character representation of the cluster.
    """
    if 0 <= cluster <= 25:
        return chr(ord('a') + cluster)
    elif 26 <= cluster <= 51:
        return chr(ord('A') + (cluster - 26))
    else:
        return chr(ord('0') + (cluster - 52))

# Function to compress contiguous clusters
def compress_clusters(clusters, include_length=False):
    """
    Compress a list of clusters into a compact string representation.

    Parameters:
        clusters (list): List of cluster IDs.
        include_length (bool): Whether to include segment lengths.

    Returns:
        str: Compressed representation of clusters.
    """
    compressed = []
    current_char = cluster_to_char(clusters[0])
    current_count = 1

    for i in range(1, len(clusters)):
        char = cluster_to_char(clusters[i])
        if char == current_char:
            current_count += 1
        else:
            if include_length:
                compressed.append(f"{current_char}[{current_count}]")
            else:
                compressed.append(current_char)
            current_char = char
            current_count = 1

    # Append the last group
    if include_length:
        compressed.append(f"{current_char}{current_count}")
    else:
        compressed.append(current_char)
    
    return "".join(compressed)


# 9, 20, 26, 41, 46, 47, 64

color_map = {
    1: "aquamarine",
    2: "black", # ok 
    3: "blue",
    4: "bluewhite",
    5: "br0",
    6: "br1",
    7: "br2",
    8: "br3",
    9: "br4", 
    10: "br5",
    11: "br6",
    12: "br7", 
    13: "br8",
    14: "br9",
    15: "brightorange",
    16: "brown",
    17: "carbon",
    18: "chartreuse",
    19: "chocolate",
    20: "cyan", 
    21: "darksalmon",
    22: "dash",
    23: "deepblue",
    24: "deepolive",
    25: "deeppurple",
    26: "deepsalmon", #"",
    27: "pink",
    28: "deepteal",
    29: "density",
    30: "dirtyviolet", 
    31: "firebrick",
    32: "forest",
    33: "gray",
    34: "green",
    35: "greencyan",
    36: "grey",
    37: "hotpink", 
    38: "hydrogen",  # ok
    39: "lightblue",
    40: "lightmagenta",
    41: "lightorange", # "wheat", # "lightorange", 
    42: "lightpink",
    43: "lightteal",
    44: "lime",
    45: "limegreen",
    46: "warmpink", #"limon",
    47: "magenta", # "violetpurple", #"magenta",
    48: "marine",
    49: "nitrogen",
    50: "olive", # ok
    51: "orange",
    52: "oxygen",
    53: "palecyan",
    54: "palegreen",
    55: "paleyellow",
    56: "pink",
    57: "purple",
    58: "purpleblue",
    59: "raspberry",
    60: "red",
    61: "ruby",
    62: "salmon",
    63: "sand",
    64: "skyblue",
    65: "slate",
    66: "smudge",
    67: "splitpea",
    68: "sulfur",
    69: "teal",
    70: "tv_blue",
    71: "tv_green",
    72: "tv_orange",
    73: "tv_red",
    74: "tv_yellow",
    75: "violet",
    76: "violetpurple",
    77: "warmpink",
    78: "wheat",
    79: "white",
    80: "yellow",
    81: "yelloworange",
}





# Function to color clusters in a PDB structure
def color_clusters(pymol_cmd, pdb_path, clusters, output_dir=None):
    """
    Color residues in a PDB file based on cluster assignments.

    Parameters:
        pymol_cmd: PyMOL command interface.
        pdb_path (str): Path to the PDB file.
        clusters (list): List of cluster IDs for residues.
        output_dir (Path): Directory to save the colored PDB file (optional).
    """
    # Load the PDB structure
    pymol_cmd.load(f"{pdb_path}")  # Ensure the PDB file is in the current directory

    # Iterate over the residues and color them according to the cluster    
    for i in range(len(clusters)):
        res_index = i + 1  # PyMOL uses 1-based indexing for residues
        cluster_id = clusters[i]   # Adjust for 0-based indexing in color_map
        if 0 <= cluster_id < len(color_map):
            # Define and set the custom color in PyMOL
            pymol_cmd.color(color_map[cluster_id+1], f"resi {res_index}")  # Apply the color
    print(output_dir)
    if output_dir: 
        filepath = str(output_dir / (f'{pdb_path.name}.pdb' ) if not pdb_path.name.endswith('pdb') else pdb_path.name)
        filepath = output_dir / pdb_path.name.replace('.pdb', '.pse')
        print(filepath)
        pymol_cmd.save(filepath)

# Function to visualize and save colored PDB
def color_pdb(pdb_path, clusters, output_dir=None):
    """
    Visualize and save a PDB structure colored by clusters.

    Parameters:
        pdb_path (str): Path to the PDB file.
        clusters (list): List of cluster IDs for residues.
        output_dir (Path): Directory to save outputs (optional).
    """
    with pymol2.PyMOL() as pymol:
        color_clusters(pymol.cmd, pdb_path, clusters, output_dir)
        
        pymol.cmd.show("cartoon")
        pymol.cmd.zoom()
    
        # Render and save the image as a PNG file
        pymol.cmd.ray()  # Renders the scene to improve quality
        pymol.cmd.png(str(output_dir / f"{pdb_path.name}.png"), width=800, height=600, dpi=300)  # Save as PNG

    # Load and display the image with Matplotlib
    img = Image.open(str(output_dir / f"{pdb_path.name}.png"))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def group_consecutive_ids(sequence):
    """
    Group consecutive identical IDs in a sequence.

    Parameters:
        sequence (list): List of cluster IDs.

    Returns:
        list: List of formatted group strings.
    """
    result = []
    start_index = 0

    # Iterate through the sequence
    while start_index < len(sequence):
        current_id = sequence[start_index]
        end_index = start_index

        # Find the end of the current group of the same ID
        while end_index + 1 < len(sequence) and sequence[end_index + 1] == current_id:
            end_index += 1
        
        # Append the range and ID in the desired format
        result.append(f"{start_index + 1}-{end_index + 1}:{current_id}")
        
        # Move to the next new ID
        start_index = end_index + 1

    return result

# Function to convert positions to the specified format
def format_positions(positions, min_length=1):
    """
    Format positions into ranges based on groupings.

    Parameters:
        positions (list): List of positions to format.
        min_length (int): Minimum range length to include.

    Returns:
        list: List of formatted ranges.
    """
    ranges = []
    for k, g in groupby(enumerate(positions), lambda i: i[0] - i[1]):
        group = list(map(lambda x: x[1] + 1, g))  # Adding 1 for 1-based index
        if len(group) < min_length: 
            continue
        if len(group) == 1:
            ranges.append(str(group[0]))
        else:
            ranges.append(f"{group[0]}-{group[-1]}")
    return '+'.join(ranges)

def sequence_to_dict(sequence, min_length=1):
    """
    Create a dictionary where keys are unique elements in the sequence,
    and values are formatted position lists.

    Parameters:
        sequence (list): Input sequence.
        min_length (int): Minimum segment length to include in the result.

    Returns:
        dict: Dictionary with formatted positions for each unique element.
    """
    pos_dict = {}
    for index, value in enumerate(sequence):
        pos_dict.setdefault(value, []).append(index)

    # Format positions and filter clusters
    result_dict = {
        str(key): format_positions(value, min_length)
        for key, value in pos_dict.items()
    }
    return {cluster: pos for cluster, pos in result_dict.items() if pos != ''}


# cluster_dict = df_clusters.set_index('id')['clusters'].to_dict()
# cluster_representation = {pdb: sequence_to_dict(rep.split(',')) for pdb, rep in cluster_dict.items()}
# cluster_representation

def continuous_segments(positions, min_length=1):
    """
    Calculate lengths of continuous segments in a list of positions.

    Parameters:
        positions (list): Sorted list of positions.
        min_length (int): Minimum segment length to include.

    Returns:
        list: List of segment lengths.
    """
    ranges = []
    for _, group in groupby(enumerate(positions), lambda i: i[0] - i[1]):
        segment = list(map(lambda x: x[1] + 1, group))  # Convert to 1-based indexing
        if len(segment) >= min_length:
            ranges.append(len(segment))
    return ranges

def sequence_to_cts_seg(sequence, min_length=1):
    """
    Generate continuous segment lengths for each unique element in the sequence.

    Parameters:
        sequence (list): Input sequence.
        min_length (int): Minimum segment length to include.

    Returns:
        list: Continuous segment lengths.
    """
    pos_dict = {}
    for index, value in enumerate(sequence):
        pos_dict.setdefault(value, []).append(index)

    return [length for value in pos_dict.values() for length in continuous_segments(value, min_length)]

def parse_positions(positions_str):
    """
    Parse position strings into a list of individual positions.

    Parameters:
        positions_str (str): Position string in the format "1-3+5".

    Returns:
        list: List of individual positions.
    """
    positions = []
    for rng in positions_str.split('+'):
        if '-' in rng:
            start, end = map(int, rng.split('-'))
            positions.extend(range(start, end + 1))
        else:
            positions.append(int(rng))
    return positions

def compute_cluster_freq_and_length(cluster_representation):
    """
    Compute frequency and lengths of clusters in a representation.

    Parameters:
        cluster_representation (dict): Mapping of PDB IDs to cluster dictionaries.

    Returns:
        tuple: DataFrame of cluster lengths, Counter of cluster frequencies.
    """
    cluster_lengths = []
    cluster_frequencies = Counter()
    unique_clusters_per_structure = {}
    all_clusters = set()

    for pdb_id, clusters in cluster_representation.items():
        unique_clusters_per_structure[pdb_id] = set(clusters.keys())
        all_clusters.update(clusters.keys())

        for cluster, positions_str in clusters.items():
            positions = parse_positions(positions_str)
            cluster_frequencies[cluster] += 1
            cluster_lengths.append((cluster, len(positions)))

    df_lengths = pd.DataFrame(cluster_lengths, columns=['Cluster', 'Length'])

    unique_clusters = len(all_clusters)
    common_clusters = [cluster for cluster, count in cluster_frequencies.items() if count > 1]

    print("Total number of unique clusters:", unique_clusters)
    print("Unique clusters:", set(all_clusters).difference(set(common_clusters)))
    print("Cluster frequency distribution:\n", cluster_frequencies)
    print("Common clusters (appear in multiple structures):", common_clusters)
# cluster_dict_esm = df_clusters_esm.set_index('id')['clusters'].to_dict()
# all_cnts_lengths = [l for rep in cluster_dict.values() for l in sequence_to_cts_seg(rep.split(','))] 
# all_cnts_lengths_esm = [l for rep in cluster_dict_esm.values() for l in sequence_to_cts_seg(rep.split(','))] 
# len(all_cnts_lengths_esm)



def generate_cluster_boundaries(protein_length: int, sample_lengths: List[int]):
    """
    Generate random cluster boundaries.

    Parameters:
        protein_lengths (list): List of protein lengths.
        lengths (int): Possible segment lengths.

    Returns:
        list: random cluster sequences for a protein.
    """
    cluster_id = 0
    remaining_length = protein_length
    cluster_sequence = []

    while remaining_length > 0:
        sampled_length = np.random.choice(sample_lengths)
        sampled_length = min(sampled_length, remaining_length)
        cluster_sequence.extend([cluster_id] * sampled_length)
        cluster_id += 1
        remaining_length -= sampled_length

    return cluster_sequence

# Example data
# np.random.seed(42)
# protein_lengths = df_clusters['clusters_length'].tolist()

# # Generate random clusters using boundaries
# random_clusters_boundaries = generate_cluster_boundaries(protein_lengths, all_cnts_lengths)
# random_clusters_boundaries

def compute_iou(clusters, annotations):
    """
    Compute Intersection over Union (IoU) between clusters and annotations.

    Parameters:
        clusters (list): Predicted cluster assignments.
        annotations (list): List of ground-truth annotation ranges.

    Returns:
        tuple: IoU chain score and best matching cluster domains.
    """
    ground_truth_domains = [set(range(start, end + 1)) for start, end in annotations]
    unique_clusters = set(clusters)
    predicted_domains = {
        cluster_id: {i for i, c in enumerate(clusters) if c == cluster_id}
        for cluster_id in unique_clusters
    }

    matched_clusters = []
    results = []
    coverage_stats = []
    for gt_domain in ground_truth_domains:
        best_score = 0
        best_cluster = None

        for cluster_id, pred_domain in predicted_domains.items():
            intersection = len(gt_domain & pred_domain)
            union = len(gt_domain | pred_domain)
            iou = intersection / union if union > 0 else 0
            dice = 2 * intersection / (len(gt_domain) + len(pred_domain)) if (len(gt_domain) + len(pred_domain)) > 0 else 0
            annotation_coverage = intersection / len(gt_domain) if gt_domain else 0
            cluster_coverage = intersection / len(pred_domain) if pred_domain else 0
            
            # Combined score with coverage penalty
            coverage_score = (annotation_coverage + cluster_coverage) / 2
            combined_score = (iou + coverage_score) / 2
            if combined_score > best_score:
                best_score = combined_score
                # best_score = dice
                best_cluster = cluster_id
                best_stats = {
                    'iou': iou,
                    'annotation_coverage': annotation_coverage,
                    'cluster_coverage': cluster_coverage,
                    'combined_score': combined_score
                }
        
        if best_score > 0:
            weight = len(gt_domain) / sum(len(gt) for gt in ground_truth_domains)
            results.append(best_score * weight)
            matched_clusters.append(best_cluster)
            coverage_stats.append(best_stats)
        else:
            matched_clusters.append(None)
            coverage_stats.append(None)
    
    final_score = sum(results)
    return final_score, matched_clusters, coverage_stats


# Example usage
# cluster_sample = [26, 26, 36, 60, 26, 44, 44, 44, 44, 44]
# annotations = [(4, 7)]  # Example annotation for indices 4 to 7
# iou_chain, best_match = compute_iou(cluster_sample, annotations)
# print(f"IoU_chain score: {iou_chain}, best matching cluster: {best_match}")

def find_cluster_assignments(
    batch, 
    cluster_assignments_matrix, 
    cluster_confidence_matrix, 
    cluster_entropies,
    cluster_diffs, 
    level=0):
    """
    Extracts cluster assignments and related metrics from a batch of data.

    Parameters:
        batch (list): A list of batch elements, each containing residue data.
        cluster_assignments_matrix (torch.Tensor): A 2D tensor of shape (B, N), where B is the batch size 
                                                   and N is the maximum number of residues per batch element.
        cluster_confidence_matrix (torch.Tensor): A 2D tensor of confidence values corresponding to cluster assignments.
        cluster_entropies (torch.Tensor, optional): A 2D tensor of entropy values for the clusters (default: None).
        cluster_diffs (torch.Tensor, optional): A 2D tensor of top probability differences for the clusters (default: None).
        level (int, optional): The hierarchy level for residue lengths. If non-zero, uses a fixed residue length (N) 
                              for all batches; otherwise, dynamically determines residue lengths based on the batch data.

    Returns:
        tuple: Contains the following lists for each batch element:
            - cluster_assignments: List of assigned cluster IDs for each residue.
            - cluster_confidences: List of confidence values for each residue.
            - cluster_entropies_list: List of entropy values for each residue.
            - cluster_diffs_list: List of top probability differences for each residue.
    """
    # Get the dimensions of the cluster assignments matrix
    batch_size, max_nodes = cluster_assignments_matrix.size()
    cluster_assignments = []  # Stores cluster assignments for all batch elements
    cluster_confidences = []
    cluster_entropies_list = []
    cluster_diffs_list = []
    # Determine residue lengths based on the provided level
    if level != 0:
        # Use the maximum number of nodes for all batch elements
        residue_lengths = [max_nodes] * batch_size
    else:
        # Dynamically determine residue lengths based on the batch
        residue_lengths = [len(batch[b].residues) for b in range(batch_size)]

    # Iterate over each batch element
    for b in range(batch_size):
        batch_cluster_assignments = []  # Stores cluster assignments for the current batch element
        batch_cluster_confidence = []
        batch_entropies = []
        batch_diffs = [] 
        # Process nodes up to the determined residue length for the batch element
        for n in range(residue_lengths[b]):
            cluster_assignment = cluster_assignments_matrix[b, n].item()  # Get cluster assignment
            # Only include valid assignments (ignore placeholder -1)
            if cluster_assignment != -1:
                batch_cluster_assignments.append(cluster_assignment)
                batch_cluster_confidence.append(cluster_confidence_matrix[b, n].item())
                batch_entropies.append(cluster_entropies[b, n].item())
                batch_diffs.append(cluster_diffs[b, n].item())

        # Append the current batch's assignments to the overall list
        cluster_assignments.append(batch_cluster_assignments)
        cluster_confidences.append(batch_cluster_confidence)
        cluster_entropies_list.append(batch_entropies)
        cluster_diffs_list.append(batch_diffs)

    return cluster_assignments, cluster_confidences, cluster_entropies_list, cluster_diffs_list

def find_communities(data):
    data_g = to_networkx(data)
    communities = nx.community.louvain_communities(data_g, seed=123)
    node_to_community = {}
    for community_id, community in enumerate(communities, start=1):  # Start IDs from 1
        for node in community:
            node_to_community[node] = community_id

    return [node_to_community[node] for node in sorted(data_g.nodes())]
