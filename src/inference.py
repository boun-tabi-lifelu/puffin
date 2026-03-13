import collections
import os
import pathlib
import rootutils
import functools
import pandas as pd
import hydra
import lightning as L
import omegaconf
import copy
import torch
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from loguru import logger as log
from tqdm import tqdm
from pathlib import Path 

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from proteinworkshop import (
    register_custom_omegaconf_resolvers,
)
from proteinworkshop.features.sequence_features import amino_acid_one_hot

from cafaeval.evaluation import cafa_eval, write_results
from captum.attr import IntegratedGradients, LayerIntegratedGradients, LayerGradCam, GuidedGradCam

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import (
    extras,
)
from src import (
    register_custom_omegaconf_resolvers as src_register_custom_omegaconf_resolvers,
)
from src.models.dual_model import DualModel
from src.models.unsupervised_model import UnsupervisedModel
from src.models.contrastive_model import ContrastiveModel
from src.utils.cluster_utils import find_cluster_assignments, color_clusters, color_pdb
from src.utils.model_utils import load_model
from src.data.singleprotein import SingleProteinDataset
from src.data import BioLiPManager, BioLiPNRManager

# from proteinworkshop.datasets.base import ProteinDataset
from torch_geometric.data import Dataset
from torch_geometric.utils import unbatch
from torch_geometric import transforms as T
from torch_geometric.data import Data, Dataset
from loguru import logger 
from graphein.protein.tensor.io import to_dataframe
from graphein.protein.tensor.dataloader import ProteinDataLoader
from graphein.protein.tensor.io import protein_to_pyg
from graphein.protein.utils import (
    download_pdb_multiprocessing,
    get_obsolete_mapping,
)
from graphein import verbose


verbose(False)


STANDARD_AMINO_ACID_MAPPING_3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PYL": "O",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "SEC": "U",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
}

def normalize_attributions(scores): 
    scores = scores - scores.min()
    scores = scores / scores.max()
    scores = scores * 100
    return scores.cpu()

def compute_labels(binding_site_residues, num_data_points):
        """
        Compute binary labels for binding/non-binding based on binding site information.
        """
        bind_index = [int(i[1:])-1 for i in binding_site_residues.split()]
        labels = [1 if i in bind_index else 0 for i in range(num_data_points)]
        return labels


def compute_roc_auc(attribution_scores, binding_site_residues):
    """
    Compute the ROC AUC score.
    """
    labels = compute_labels(binding_site_residues, len(attribution_scores))
    return roc_auc_score(labels, attribution_scores), labels


def plot_attribution_and_binding(attribution_scores, binding_labels, score_type, output_dir):
    """
    Plot attribution scores and binding indices.

    Args:
        attribution_scores (list or np.array): Attribution scores for residues.
        binding_labels (list): Indices of binding residues.
        output_dir (pathlib.Path): Directory to save the plot.
    """

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(attribution_scores)), attribution_scores, label=f'{score_type} Attribution Scores', color='orange')
    binding_marks = {ix:attribution_scores[ix] for ix, mask in enumerate(binding_labels) if mask == 1}
    plt.scatter(list(binding_marks.keys()),list(binding_marks.values()) ,
                color='blue', label='Binding Residues', zorder=3)
    
    plt.xlabel('Residue Index')
    plt.ylabel('Attribution Score')
    plt.title(f'{score_type} Attribution Scores and Binding Indices')
    plt.legend()
    plt.grid()

    # Save the plot
    output_path = output_dir / f'attribution_binding_{score_type}.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot for {score_type} attribution and binding indices saved at: {output_path}")




def plot_and_save_roc_curve(labels, scores, roc_auc, score_type, output_dir):
    """
    Plot and save the ROC curve.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {score_type} Scores')
    plt.legend(loc='lower right')
    plt.grid()
    
    output_path = output_dir / f'roc_curve_{score_type}.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"ROC curve for {score_type} saved at: {output_path}")

# TODO: fix color with to_dataframe

def inference(cfg: omegaconf.DictConfig):
    assert cfg.ckpt_path, "No checkpoint path provided."
    assert (
        cfg.output_dir
    ), "No output directory for attributed PDB files provided."


    L.seed_everything(cfg.seed)

    dataset = SingleProteinDataset(
        cfg.pdb_path, 
        cfg.annotation_data_path, 
        cfg.split)
    
    dataloader = ProteinDataLoader(
            dataset._get_dataset(),
            batch_size=32,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )

    model = load_model(cfg, batch=next(iter(dataloader)))


    output_dir = pathlib.Path(cfg.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log.info(f"Performing clustering for {cfg.pdb_path}")
    cluster_assignments = []
    for batch in tqdm(dataloader):
        batch = batch.cuda()
        batch = model.featurise(batch)
        batch_items = batch.to_data_list()
        node_features = batch.x 
        with torch.no_grad():
            output = model.encoder.forward(batch, return_clusters=True)
            clusters = output["clusters"]
            confidence = output["confidence"]
            entropy = output["entropy"]
            top_difference = output["top_difference"]

            prediction = model.forward(batch)[0]["graph_label"]

        predictions = []
        ground_truth = []
        for bid, item in tqdm(enumerate(batch_items)):
            for label, lid in dataset.label2id.items():
                predictions.append({'EntryID': item.id.upper(), 'term': label, 'score': prediction[bid][lid].cpu().item()})
                # item.graph_y is string when no label is present for the structure
                if type(item.graph_y) != str and lid in item.graph_y: 
                    ground_truth.append({'EntryID': item.id.upper(), 'term': label})
        # Target ID, term ID, score columns, tsv
        df = pd.DataFrame(predictions)
        gt = pd.DataFrame(ground_truth)

        save_path = output_dir / 'evaluation' 
        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(save_path / "predictions.tsv"), index=False, header=False, sep="\t")
        log.info(f"Saved predictions to {save_path}")

        if len(ground_truth) > 0: 
            gt.to_csv(str(save_path / "ground_truth.tsv"), index=False, header=False, sep="\t")

            # CAFA5 
            # IA.txt, no_orphans=False, norm='cafa', prop='fill', max_terms=500, th_step=0.001
            # Default
            # ia=None, no_orphans=False, norm='cafa', prop='max', max_terms=None, th_step=0.01
       
            # TODO: 2FGE-A
            # deepest = np.where(np.sum(matrix[:, order], axis=0) > 0)[0][0]
            # IndexError: index 0 is out of bounds for axis 0 with size 0
            #eval_results = cafa_eval(cfg.obo_path, save_path, save_path / "ground_truth.tsv", ia=cfg.ia, no_orphans=cfg.no_orphans, 
            #                            norm=cfg.norm, prop=cfg.prop, max_terms=cfg.max_terms, th_step=cfg.th_step)
            # write_results(*eval_results, out_dir=save_path)


        for cluster_, confidence_, entropy_, top_diff_ in zip(clusters, confidence, entropy, top_difference):
            cluster_assigns, cluster_confidences, cluster_entropy, cluster_top_diff = find_cluster_assignments(
                batch_items, cluster_.cpu(), confidence_.cpu(), entropy_.cpu(), top_diff_.cpu()
            )
            for item, cluster_assign, cluster_conf, cluster_ent, cluster_diff in zip(batch_items, cluster_assigns, cluster_confidences, cluster_entropy, cluster_top_diff):
                cluster_assignments.append({
                    'id': item.id.upper(),
                    'clusters': ','.join(map(str, cluster_assign)),
                    'confidence': ','.join(map(str, cluster_conf)),
                    'entropy': ','.join(map(str, cluster_ent)),
                    'top_difference': ','.join(map(str, cluster_diff))
                })

                if cfg.highlight: 
                    color_pdb(Path(cfg.pdb_path), cluster_assign, output_dir=output_dir)
         

                    # color_df = to_dataframe(
                    #    x=item.coords,
                    #    residue_types=item.residues,
                    #     chains=item.chains,
                    #     insertions=None,
                    #     b_factors=cluster_assign, 
                    #     occupancy=None,
                    #     charge=None,
                    #     alt_loc=None,
                    #     segment_id=None,
                    #     biopandas=True,
                    # )
                    # output_path = output_dir / f"{item.id}.pdb"
                    # color_df.to_pdb(str(output_path))

        df = pd.DataFrame.from_records(cluster_assignments)
        print(df)
        df.to_csv(output_dir / 'clusters.csv', index=False)
       
        if cfg.explain: 




            # Wrap forward function and pass to captum
            def _forward_wrapper_ig(
                model,
                node_feats,
                batch,
            ):
                """Wrapper function around forward pass.

                Sets node features to the provided node features and returns the
                model output for the specified output.

                The node feature update is necessary to set the interpolated features
                from IntegratedGradients.
                """
                batch.x = node_feats  # Update node features
                return model.forward(batch)[0]["graph_label"]
            
            def _forward_wrapper_gc(
                model,
                structural_features,
                esm_embeddings,
                batch,
            ):  
                
                batch.x = structural_features     
                # Note: Commented out for subclustgo without embeddings  
                batch.esm_embeddings = esm_embeddings

                return model.forward(batch)[0]["graph_label"]

            if cfg.units == 'residue': 
                fwd = functools.partial(_forward_wrapper_gc, model)
                ig = IntegratedGradients(fwd)
                ig = LayerGradCam(fwd, model.encoder.conv1) # convs[1]) # convs[0], convs[1], conv2
            # elif cfg.units == 'cluster':
            #     # AttributeError: 'Tensor' object has no attribute 'x'
            #     fwd = functools.partial(_forward_wrapper_ig, model)
            #     ig = LayerIntegratedGradients(model, model.encoder.conv2)

     
            else:
                raise ValueError(f"Unit {cfg.units} not recognised.") 
            # fwd = functools.partial(_forward_wrapper_gc, model.encoder)
            # ig = GuidedGradCam(fwd, model.encoder.conv2)  # conv2 for cluster attribution

            attributions_path = output_dir / cfg.target 
            attributions_path.mkdir(parents=True, exist_ok=True)

            class_id = dataset.label2id[cfg.target]
            log.info(f'Generating explanation for {cfg.target}: {class_id}')

       
            # Note: Commented out for subclustgo without embeddings  
            # esm_embeddings = model.encoder.esm_model.esm_embed(batch)
            # baseline_structural = torch.zeros_like(node_features)
            # baseline_esm = torch.zeros_like(esm_embeddings)
            # IntegratedGradients attribute
            # attr_structural, attr_esm = ig.attribute(
            #     inputs = (node_features, esm_embeddings),
            #     baselines=(baseline_structural, baseline_esm),
            #     additional_forward_args=batch,
            #     target=class_id.item(),
            #     internal_batch_size=50,
            #     n_steps=cfg.n_steps,
            # )
            esm_embeddings = None
         

            # GradCam attribute
            attr = ig.attribute(
                inputs = (node_features, esm_embeddings),
                additional_forward_args=batch,
                target=class_id.item(),
                relu_attributions=True,
            )
            gradcam_attr = normalize_attributions(attr).detach().numpy().T # .squeeze().tolist()
            print(gradcam_attr.shape)

            biolip = BioLiPManager()

            pdb_id, pdb_chain = [dataset.pdb], [dataset.chain]
            biolip_ann = biolip.get_annotations(pdb_id, pdb_chain)
            if biolip_ann.shape[0] == 0:
                log.info('No entry found in BioLip skipping evaluation')
            else: 
                # biolip_ann = biolip_ann[biolip_ann['ligand_id'] == 'dna']
                #print(biolip_ann[['binding_site_residues', 'catalytic_site_residues']])
                # TODO: make sure that the GO term of interest appear in each row. 
                pdb_binding_site_residues = ' '.join(biolip_ann['binding_site_residues'].tolist())
                print(pdb_binding_site_residues)

                # df_biolip['binding_site'] = df_biolip['binding_site_residues'].apply(lambda s: ''.join([i[0] for i in s.split()])) 
                # df_biolip['binding_site_start'] = df_biolip['binding_site_residues'].apply(lambda s: s.split()[0][1:]) 

                # print(df_biolip)
                # print(df_biolip.iloc[0]['binding_site'])
            
                roc_auc_gradcam, labels_gradcam = compute_roc_auc(gradcam_attr[0], pdb_binding_site_residues)
                print(f"ROC AUC for GradCam Conv1 scores: {roc_auc_gradcam}")

                # Plot and save the ROC curves
                plot_and_save_roc_curve(labels_gradcam, gradcam_attr[0], roc_auc_gradcam, "GradCAM", output_dir)
                plot_attribution_and_binding(gradcam_attr[0], labels_gradcam, '', output_dir)


                # IG for residue attribution
                # attribution = ig.attribute(
                #     node_features,
                #     baselines=torch.ones_like(batch.x),
                #     additional_forward_args=batch,
                #     target=class_id.item(),
                #     internal_batch_size=cfg.dataset.datamodule.batch_size,
                #     n_steps=cfg.n_steps,
                # )

                # LayerGradCam for cluster attribution
                # attribution = ig.attribute(
                #     node_features,
                #     additional_forward_args=batch,
                #     target=class_id.item(),
            
                # )
                attr_structural = attr_structural.sum(-1)
                attr_esm = attr_esm.sum(-1)
                attr_list = []
                # Unbatch and write each protein to disk
                batch = batch.cpu()
                batch_items = batch.to_data_list()

                attribution_st_scores = unbatch(attr_structural, batch.batch)
                attribution_esm_scores = unbatch(attr_esm, batch.batch)

                for elem, str_score, esm_score in tqdm(zip(batch_items, attribution_st_scores, attribution_esm_scores)):
                    # Scale score between 0-100

                    str_score = normalize_attributions(str_score)
                    esm_score = normalize_attributions(esm_score)

                    if cfg.units == 'residue': 
                        for score_type, score in [('st', str_score), ('esm', esm_score)]:
                            df = to_dataframe(
                                x=elem.coords,
                                residue_types=elem.residues,
                                chains=elem.chains,
                                insertions=None,
                                b_factors=score,  # Write attribution score in B factor column
                                occupancy=None,
                                charge=None,
                                alt_loc=None,
                                segment_id=None,
                                biopandas=True,
                            )
                            output_path = output_dir / f"{elem.id}_{cfg.units}_{score_type}.pdb"
                            df.to_pdb(str(output_path))

                        
                    attr_list.append(
                        {
                            'id': elem.id, 
                            'class_id': class_id, 
                            'residues': elem.residues, 
                            'st_attrs': ','.join([str(i) for i in str_score.tolist()]),
                            'esm_attrs': ','.join([str(i) for i in esm_score.tolist()]),
                        }
                    )
                    
            
                pd.DataFrame.from_records(attr_list).to_csv(attributions_path / f'{cfg.units}_attributions.csv')

                aa = ''.join([STANDARD_AMINO_ACID_MAPPING_3_TO_1[i] for i in attr_list[0]['residues']])
                assert aa == biolip_ann['receptor_sequence'].iloc[0]

                

                # Compute ROC AUC and plot the curves
                roc_auc_esm, labels_esm = compute_roc_auc(esm_score, pdb_binding_site_residues)
                roc_auc_str, labels_str = compute_roc_auc(str_score, pdb_binding_site_residues)
                combined_score = normalize_attributions(str_score + esm_score)
                roc_auc_esm_str, labels_esm_str = compute_roc_auc(combined_score, pdb_binding_site_residues)

                print(f"ROC AUC for ESM scores: {roc_auc_esm}")
                print(f"ROC AUC for Structural scores: {roc_auc_str}")
                print(f"ROC AUC for Combined scores: {roc_auc_esm_str}")

                # Plot and save the ROC curves
                plot_and_save_roc_curve(labels_esm, esm_score, roc_auc_esm, "ESM", output_dir)
                plot_and_save_roc_curve(labels_str, str_score, roc_auc_str, "Structural", output_dir)
                plot_and_save_roc_curve(labels_esm_str, combined_score, roc_auc_esm_str, "ESM+Structural", output_dir)

# Load hydra config from yaml files and command line arguments.
@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="inference",
)
def main(cfg: omegaconf.DictConfig) -> None:
    """Load and validate the hydra config."""
    extras(cfg)
    inference(cfg)


if __name__ == "__main__":
    
    register_custom_omegaconf_resolvers()
    src_register_custom_omegaconf_resolvers()
    main()
