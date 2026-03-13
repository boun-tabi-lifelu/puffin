import collections
import os
import pathlib
import sys
import pickle
import pandas as pd
import hydra
import lightning as L
import omegaconf
import torch
import torch.nn.functional as F
from scipy.special import expit 
from sklearn.metrics import  average_precision_score
import rootutils
import numpy as np
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import List, Union
from tqdm import tqdm
from graphein.protein.tensor.data import ProteinBatch
from graphein.protein.tensor.io import to_dataframe
from loguru import logger as log
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
from src import (
    register_custom_omegaconf_resolvers as src_register_custom_omegaconf_resolvers,
)
from src.utils.model_utils import load_model
from src.utils.eval_utils import (
    Ontology, propagate_annots, 
    compute_metrics, compute_macro_aupr, 
    stratify_aupr_values, print_stratified_results, save_stratified_results
)
from proteinworkshop import (
    register_custom_omegaconf_resolvers,
    utils,
)
from src.utils import extras
from proteinworkshop.configs import config
from proteinworkshop.models.base import BenchMarkModel
# from proteinworkshop.datasets.go import GeneOntologyDataset
from cafaeval.evaluation import cafa_eval, write_results
from pytorch_lightning import LightningDataModule

def evaluate(cfg: omegaconf.DictConfig):
    # assert cfg.ckpt_path, "No checkpoint path provided."
    #assert (
    #    cfg.output_dir
    #), "No output directory for attributed PDB files provided."


    L.seed_everything(cfg.seed)

    # dataset = GeneOntologyDataset(
    #     path='data/GeneOntology',
    #     pdb_dir='data/pdb',
    #     format='mmtf', # pdb
    #     batch_size=32,
    #     dataset_fraction=1.0,
    #     shuffle_labels=False,
    #     pin_memory=True,
    #     num_workers=8,
    #     split=cfg.split,
    #     transforms=None, # uses GOLabeler by default. 
    #     overwrite=False,
    #     in_memory=True,
    # )
    # dataset.datamodule.setup()
    # datamodule = dataset["datamodule"]
    # num_classes = 1943 for BP 320 for CC 489 for MF 
    # num_classes = dataset.datamodule.num_classes

    log.info(f"Instantiating datamodule <{cfg.dataset.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset.datamodule)

    output_dir = pathlib.Path(cfg.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    go = Ontology(cfg.obo_path, with_rels=False)
    # Retrieve all annotations, not only training.
    annotations = list(datamodule.parse_labels()[1].values())
    annotations = list(map(lambda x: (set(x) | go.get_prop_terms(x)), annotations))
    # annotations = train_df['prop_annotations'].values
    # annotations = list(map(lambda x: set(x), annotations))
    # test_annotations = test_df['prop_annotations'].values
    # test_annotations = list(map(lambda x: set(x), test_annotations))
    # go.calculate_ic(annotations + test_annotations)

    go.calculate_ic(annotations)

    dataloaders = {}
    
    splits = datamodule.test_dataset_names
    for i, split in enumerate(splits):
        dataloaders[split] = datamodule.get_test_loader(split)

    model = load_model(cfg, batch=next(iter(list(dataloaders.values())[0])))
    model.eval()
    # Iterate over batches and perform attribution
    for split, dataloader in dataloaders.items():
        log.info(f"Performing evaluation for split: {split}")

        save_path = output_dir.parent / split 
        save_path.mkdir(parents=True, exist_ok=True)
        
        go_terms = list(datamodule.label2id.keys())
        prediction_pkl = {
            'split': split,
            'ontology': datamodule.split.lower(), 
            'proteins': [], 
            'gonames': [], 
            'goterms': go_terms, 
            'Y_pred': [],
            'Y_true': [],
            'Y_pred_prop': [],
            'Y_true_prop': []
        }
        # pd.DataFrame(prediction_pkl).to_pickle(save_path / 'metadata.pkl')

        predictions = []
        ground_truth = []
        
        # pred_path = save_path / 'predictions'
        # pred_path.mkdir(parents=True, exist_ok=True)
        for batch in tqdm(dataloader):
            batch = batch.cuda()
            batch = model.featurise(batch)
            output = model.forward(batch)
            if "graph_label" in output:
                prediction = torch.sigmoid(output["graph_label"]).detach().cpu().numpy()
            else:
                # subgnnclustgo
                prediction = torch.sigmoid(output[0]["graph_label"]).detach().cpu().numpy()

            batch = batch.cpu()
            batch_items = batch.to_data_list()

            for bid, item in tqdm(enumerate(batch_items)):
                preds = prediction[bid]
                labels = item.graph_y[0].detach().cpu().numpy()
                prediction_pkl['Y_pred'].append(preds)
                prediction_pkl['Y_true'].append(labels)
                prediction_pkl['Y_pred_prop'].append(propagate_annots(preds, go, datamodule.label2id))
                prediction_pkl['Y_true_prop'].append(propagate_annots(labels, go, datamodule.label2id))
                prediction_pkl['proteins'].append(item.id)

                # pd.DataFrame(
                #     {
                #         'Y_pred': prediction[bid], 
                #         'Y_true': item.graph_y[0].cpu()
                #     }
                # ).to_pickle(pred_path / f'{item.id}.pkl') 
                    
                for label, lid in datamodule.label2id.items():
                    predictions.append({'EntryID': item.id, 'term': label, 'score': prediction[bid][lid].item()})
                    if item.graph_y[0][lid] == 1:
                        ground_truth.append({'EntryID': item.id, 'term': label})


                torch.cuda.empty_cache()
        with open(save_path / "predictions.pkl", 'wb') as f:
           pickle.dump(prediction_pkl, f)

        # macro_aupr = average_precision_score(
        #     np.stack(prediction_pkl['Y_true']), 
        #     np.stack(prediction_pkl['Y_pred'])
        # )

        macro_aupr_prop = average_precision_score(
            np.stack(prediction_pkl['Y_true_prop']), 
            np.stack(prediction_pkl['Y_pred_prop']), 
            average='macro'
        )
        print(macro_aupr_prop)
        macro_apr, aupr_values = compute_macro_aupr(
            np.stack(prediction_pkl['Y_true_prop']), 
            np.stack(prediction_pkl['Y_pred_prop'])
        )

        # levels = [go.get_level(term) for term in go_terms]
        # depths = [go.get_depth(term) for term in go_terms]

        # threshold = 4 
        # results = stratify_aupr_values(aupr_values, go_terms, go, threshold=4)
        # print_stratified_results(results)
        
        # Target ID, term ID, score columns, tsv

        df = pd.DataFrame(predictions)
        gt = pd.DataFrame(ground_truth)

        with open(save_path / 'evaluation_best_aupr.tsv', 'w') as f: 
            f.write(f'aupr\n{macro_aupr_prop}')

        with open(save_path / 'evaluation_best_apr.tsv', 'w') as f: 
            f.write(f'apr\n{macro_apr}')

        # save_stratified_results(results, save_path)

        cafaeval_path = output_dir.parent / split / 'cafaeval'
        cafaeval_path.mkdir(parents=True, exist_ok=True)

        df.to_csv(str(cafaeval_path / "predictions.tsv"), index=False, header=False, sep="\t")
        gt.to_csv(str(save_path / "ground_truth.tsv"), index=False, header=False, sep="\t")


        log.info(f"Saved predictions to {cafaeval_path}")

        # CAFA5 Evaluation
        # cafaeval go-basic.obo prediction_dir test_terms.tsv -ia IA.txt -prop fill -norm cafa -th_step 0.001 -max_terms 500

        eval_results = cafa_eval(cfg.obo_path, cafaeval_path, save_path / "ground_truth.tsv", ia=cfg.ia, no_orphans=cfg.no_orphans, 
                                norm=cfg.norm, prop=cfg.prop, max_terms=cfg.max_terms, th_step=cfg.th_step)
        # CAFA5 
        # IA.txt, no_orphans=False, norm='cafa', prop='fill', max_terms=500, th_step=0.001

        # Default
        # ia=None, no_orphans=False, norm='cafa', prop='max', max_terms=None, th_step=0.01

        # Write results to disk
        write_results(*eval_results, out_dir=save_path)
        


        # DeepGO variant evaluation 

        # test_df = datamodule.parse_dataset("testing", split)
        # test_df["proteins"] = test_df.apply(lambda r: f"{r['pdb'].lower()}_{r['chain']}", axis=1)
        # test_df['exp_annotations'] = test_df['label'].apply(lambda l: [datamodule.id2label[i.item()] for i in l])
        
        # test_df['prop_annotations'] =  test_df['exp_annotations'].apply(lambda x: (set(x) | go.get_prop_terms(x)))
        # test_df = test_df.set_index('proteins').reindex(prediction_pkl['proteins']).reset_index()

    
        # prop_preds = [propagate_annots(pred, go, datamodule.label2id) for pred in prediction_pkl['Y_pred']]
        # eval_preds = np.stack(prop_preds, axis=0)
        # fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic, fmax_spec_match = compute_metrics(
        #     test_df, go, datamodule.label2id, list(datamodule.label2id.keys()), datamodule.split.lower(), eval_preds)
        
        
        # print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}, spec: {fmax_spec_match}')
        # print(f'WFmax: {wfmax:0.3f}, threshold: {wtmax}')
        # print(f'AUC: {avg_auc:0.3f}')
        # print(f'AUPR: {aupr:0.3f}')
        # print(f'AVGIC: {avgic:0.3f}')



# Load hydra config from yaml files and command line arguments.
@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="eval",
)
def _main(cfg: omegaconf.DictConfig) -> None:
    """Load and validate the hydra config."""
    extras(cfg)
    evaluate(cfg)


def _script_main(args: List[str]) -> None:
    """
    Provides an entry point for the script dispatcher.

    Sets the sys.argv to the provided args and calls the main train function.
    """
    sys.argv = args
    register_custom_omegaconf_resolvers()
    _main()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    register_custom_omegaconf_resolvers()
    _main()  # type: ignore
