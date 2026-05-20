#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import shutil
import argparse

# 防止多进程与CUDA冲突
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

sys.path.append('.')

import torch
import numpy as np
from scipy import spatial
import torch.utils.tensorboard
from easydict import EasyDict
from rdkit import Chem
from tqdm import tqdm

from torch_scatter import scatter_sum
from torch_geometric.data import Batch
from models.model import DiffGui
from models.bond_predictor import BondPredictor
from utils.sample_utils import seperate_outputs
from torch_geometric.transforms import Compose
from utils.evaluation.atom_num_config import CONFIG
from utils.data import PDBProtein, parse_lig_file
from utils.dataset import to_torch_dict, get_dataset, ProteinLigandData
from utils.evaluation import scoring_func
from utils.evaluation.docking_vina import VinaDockingTask
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *

# ✅ 引入批量毒性预测器
from utils.ensemble_toxicity_predictor import EnsembleToxicityPredictor


def print_pool_status(pool, logger):
    logger.info('[Pool] Finished %d | Failed %d' % (
        len(pool.finished), len(pool.failed)
    ))


def get_pocket_size(pocket_pos):
    aa_dist = spatial.distance.pdist(pocket_pos, metric="euclidean")
    aa_dist_sort = np.sort(aa_dist)[::-1]
    return np.median(aa_dist_sort[:10])


def get_bin_idx(pocket_size):
    bounds = CONFIG["bounds"]
    for i in range(len(bounds)):
        if bounds[i] > pocket_size:
            return i
    return len(bounds)


def sample_atom_num(pocket_size):
    bin_idx = get_bin_idx(pocket_size)
    num_atom_list, prob_list = CONFIG["bins"][bin_idx]
    atom_num = np.random.choice(num_atom_list, p=prob_list)
    return atom_num


def pdb_to_pocket(pocket_pdb_path, ligand_sdf_path, frag_sdf_path):
    pocket_dict = PDBProtein(pocket_pdb_path).to_dict_atom()
    if ligand_sdf_path != 'None':
        ligand_dict = parse_lig_file(ligand_sdf_path)
    else:
        ligand_dict = {
            "element": torch.empty([0, ], dtype=torch.long),
            "hybridization": torch.empty([0, ], dtype=torch.long),
            "pos": torch.empty([0, 3], dtype=torch.float),
            "bond_index": torch.empty([2, 0], dtype=torch.long),
            "bond_type": torch.empty([0, ], dtype=torch.long),
            "atom_feature": torch.empty([0, 8], dtype=torch.float),
        }
    if frag_sdf_path != 'None':
        frag_dict = parse_lig_file(frag_sdf_path)
        data = ProteinLigandData.protein_ligand_dicts(
            protein_dict=to_torch_dict(pocket_dict),
            ligand_dict=to_torch_dict(ligand_dict),
            frag_dict=to_torch_dict(frag_dict)
        )
    else:
        data = ProteinLigandData.protein_ligand_dicts(
            protein_dict=to_torch_dict(pocket_dict),
            ligand_dict=to_torch_dict(ligand_dict)
        )
    return data


def main(args):
    WEIGHT_SA = 1.0
    WEIGHT_QED = 1.0
    WEIGHT_TOX = 1.0

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed + np.sum([ord(s) for s in args.outdir]))

    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    train_config = ckpt['config']

    # Logging
    log_root = args.outdir.replace('outputs', 'outputs_vscode') if sys.argv[0].startswith('/data') else args.outdir
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    logger = get_logger('sample', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # 初始化批量毒性预测器
    try:
        logger.info("正在初始化批量毒性预测器...")
        toxicity_predictor = EnsembleToxicityPredictor(
            script_path=args.tox_script_path,
            workdir=args.tox_workdir
        )
        logger.info("毒性预测器初始化成功。")
    except Exception as e:
        logger.error(f"无法初始化毒性预测器: {e}")
        toxicity_predictor = None

    # Transform
    logger.info('Loading data placeholder...')
    ligand_atom_mode = ckpt["config"].data.transform.ligand_atom_mode
    if config.model.gen_mode == 'denovo':
        featurizer = FeatureComplex(ligand_atom_mode, sample=config.sample.sample)
    else:
        featurizer = FeatureComplexWithFrag(ligand_atom_mode, sample=config.sample.sample)
    transform = Compose([featurizer])
    add_edge = getattr(config.sample, 'add_edge', None)

    # Model
    logger.info('Loading diffusion model...')
    model = DiffGui(
        config=train_config.model,
        protein_node_types=featurizer.protein_feat_dim,
        ligand_node_types=featurizer.atom_feat_dim,
        num_edge_types=featurizer.bond_feat_dim,
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # ==========================================
    # ✅ 只有 4 个特征：sa, qed, aff, toxicity
    # ==========================================
    sa = torch.tensor([float(config.model.sa)], device=args.device).unsqueeze(-1)
    qed = torch.tensor([float(config.model.qed)], device=args.device).unsqueeze(-1)
    aff = torch.tensor([float(config.model.aff)], device=args.device).unsqueeze(-1)
    toxicity = torch.tensor([float(config.model.toxicity)], device=args.device).unsqueeze(-1)

    # 基础标签形状: [1, 4]
    base_lab = torch.cat((sa, qed, aff, toxicity), dim=1)

    # Bond predictor and guidance
    if 'bond_predictor' in config:
        logger.info('Building bond predictor...')
        ckpt_bond = torch.load(config.bond_predictor, map_location=args.device)
        bond_predictor = BondPredictor(
            config=ckpt_bond['config']['model'],
            protein_node_types=featurizer.protein_feat_dim,
            ligand_node_types=featurizer.atom_feat_dim,
            num_edge_types=featurizer.bond_feat_dim
        ).to(args.device)
        bond_predictor.load_state_dict(ckpt_bond['model'])
        bond_predictor.eval()
    else:
        bond_predictor = None

    guidance = config.sample.guidance if 'guidance' in config.sample else None

    # Load data
    if config.sample.mode == 'pocket':
        data = pdb_to_pocket(config.model.target, config.model.ligand, config.model.frag)
        data = transform(data)
        data_list = [data]
    elif config.sample.mode == 'test':
        dataset, subsets = get_dataset(config=config.data, transform=transform)
        data_list = subsets['test']
        logger.info(f'Test dataset: {len(data_list)}.')
    else:
        raise NotImplementedError('Sample mode should be pocket or test!')

    # Sampling
    for i in tqdm(range(len(data_list)), desc='Sample'):
        data = data_list[i]

        if config.sample.mode == 'pocket':
            name = config.model.target.split('/')[1].split('_')[0]
        elif config.sample.mode == 'test':
            if config.data.dataset == 'pdbbind':
                name = data.protein_filename.split('/')[0]
            elif config.data.dataset == 'crossdocked':
                name = data.protein_filename.split('.')[0]

        pool = EasyDict({'failed': [], 'finished': []})
        mol_list = []

        while len(pool.finished) < config.sample.num_mols:
            if len(pool.failed) > 3 * (config.sample.num_mols):
                logger.info('Too many failed molecules. Stop sampling.')
                break

            batch_size = args.batch_size if args.batch_size > 0 else config.sample.batch_size
            n_graphs = min(batch_size, (config.sample.num_mols - len(pool.finished)) * 2)
            batch = Batch.from_data_list([data.clone() for _ in range(n_graphs)],
                                         follow_batch=featurizer.follow_batch).to(args.device)

            if config.sample.sample_method == "priori":
                pocket_size = get_pocket_size(batch.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [sample_atom_num(pocket_size).astype(int) for _ in range(n_graphs)]
            elif config.sample.sample_method == "range":
                ligand_num_atoms = np.random.normal(24.923465, 5.516291, size=n_graphs).astype('int64')
            elif config.sample.sample_method == "ref":
                ligand_batch = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(ligand_batch), ligand_batch, dim=0).tolist()

            if config.model.gen_mode != 'denovo':
                frag_batch = batch.frag_element_batch
                frag_num_atoms = scatter_sum(torch.ones_like(frag_batch), frag_batch, dim=0).tolist()
                if not all(l > f for l, f in zip(ligand_num_atoms, frag_num_atoms)):
                    continue

            batch_holder = make_data_placeholder(n_nodes_list=ligand_num_atoms, device=args.device)
            batch_node, halfedge_index, batch_halfedge = batch_holder['batch_node'], batch_holder['halfedge_index'], \
            batch_holder['batch_halfedge']

            # ==========================================
            # ✅ 动态调整特征向量的行数，使其匹配 n_graphs
            # ==========================================
            current_batch_lab = torch.tensor([list(base_lab[0]) for _ in range(n_graphs)]).to(args.device)

            # Inference
            if config.model.gen_mode == 'denovo':
                outputs = model.sample(
                    n_graphs=n_graphs,
                    protein_node=batch.protein_atom_feat.float(),
                    protein_pos=batch.protein_pos,
                    protein_batch=batch.protein_element_batch,
                    ligand_batch=batch_node,
                    halfedge_index=halfedge_index,
                    halfedge_batch=batch_halfedge,
                    batch_lab=current_batch_lab,
                    gui_strength=config.sample.gui_strength,
                    bond_predictor=bond_predictor,
                    guidance=guidance,
                )
            elif config.model.gen_mode in ('frag_cond', 'frag_diff'):
                outputs = model.sample_frag(
                    n_graphs=n_graphs,
                    protein_node=batch.protein_atom_feat.float(),
                    protein_pos=batch.protein_pos,
                    protein_batch=batch.protein_element_batch,
                    frag_node=batch.frag_atom_feat_full,
                    frag_pos=batch.frag_pos,
                    frag_batch=batch.frag_element_batch,
                    frag_halfedge_type=batch.frag_halfedge_type,
                    frag_halfedge_index=batch.frag_halfedge_index,
                    frag_halfedge_batch=batch.frag_halfedge_type_batch,
                    ligand_batch=batch_node,
                    halfedge_index=halfedge_index,
                    halfedge_batch=batch_halfedge,
                    batch_lab=current_batch_lab,
                    gui_strength=config.sample.gui_strength,
                    bond_predictor=bond_predictor,
                    guidance=guidance,
                    gen_mode=config.model.gen_mode
                )

            outputs = {key: [v.cpu().numpy() for v in value] for key, value in outputs.items()}
            batch_node, halfedge_index, batch_halfedge = batch_node.cpu().numpy(), halfedge_index.cpu().numpy(), batch_halfedge.cpu().numpy()

            try:
                output_list = seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
            except Exception as e:
                logger.info(f'Separate results error: {e}')
                continue

            # --- 批量提取有效分子 ---
            valid_mol_infos = []
            smiles_batch = []

            for output_mol in output_list:
                try:
                    mol_info = featurizer.decode_output(
                        pred_node=output_mol['pred'][0],
                        pred_pos=output_mol['pred'][1],
                        pred_halfedge=output_mol['pred'][2],
                        halfedge_index=output_mol['halfedge_index'],
                    )

                    if add_edge == 'openbabel':
                        del mol_info['bond_index']
                        del mol_info['bond_type']
                        del mol_info['bond_prob']

                    rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
                    smiles = Chem.MolToSmiles(rdmol)
                    contain_B = re.search(r'B(?![rR]\b)', smiles)

                    if '.' in smiles or contain_B:
                        pool.failed.append(mol_info)
                        continue

                    mol_info['rdmol'] = rdmol
                    mol_info['smiles'] = smiles
                    mol_info['output_mol'] = output_mol

                    valid_mol_infos.append(mol_info)
                    smiles_batch.append(smiles)
                except Exception as e:
                    pool.failed.append(locals().get('mol_info', {}))
                    continue

            if not smiles_batch:
                continue

            # --- 批量预测毒性 ---
            tox_scores = []
            if toxicity_predictor is not None:
                tox_scores = toxicity_predictor.predict_batch(smiles_batch)
            else:
                tox_scores = [0.0] * len(smiles_batch)

            # --- 评估、对接与计算总分 ---
            gen_list = []
            for mol_info, tox_score in zip(valid_mol_infos, tox_scores):
                if tox_score is None:
                    pool.failed.append(mol_info)
                    continue

                try:
                    rdmol = mol_info['rdmol']
                    smiles = mol_info['smiles']
                    output_mol = mol_info['output_mol']

                    chem_results = scoring_func.get_chem(rdmol)

                    if config.sample.mode == 'pocket':
                        vina_task = VinaDockingTask.from_generated_mol(rdmol, config.model.target)
                    elif config.sample.mode == 'test':
                        if config.data.dataset == 'pdbbind':
                            protein_fn = os.path.join(os.path.dirname(data.protein_filename),
                                                      os.path.basename(data.protein_filename)[:4] + '_protein.pdb')
                        elif config.data.dataset == 'crossdocked':
                            protein_fn = os.path.join(os.path.dirname(data.protein_filename),
                                                      os.path.basename(data.protein_filename)[:10] + '.pdb')
                        vina_task = VinaDockingTask.from_generated_mol(rdmol,
                                                                       os.path.join(config.data.protein_root,
                                                                                    protein_fn))

                    vina_score = vina_task.run(mode='score_only', exhaustiveness=16)

                    mol_info['sa'] = chem_results['sa']
                    mol_info['qed'] = chem_results['qed']
                    mol_info['vina_score'] = vina_score[0]['affinity']
                    mol_info['toxicity'] = tox_score

                    mol_info['mol_score'] = (
                            float(mol_info['vina_score']) -
                            (WEIGHT_SA * float(mol_info['sa'])) -
                            (WEIGHT_QED * float(mol_info['qed'])) +
                            (WEIGHT_TOX * float(mol_info['toxicity']))
                    )

                    p_save_traj = np.random.rand()
                    if p_save_traj < config.sample.save_traj_prob:
                        traj_info = [featurizer.decode_output(
                            pred_node=output_mol['traj'][0][t],
                            pred_pos=output_mol['traj'][1][t],
                            pred_halfedge=output_mol['traj'][2][t],
                            halfedge_index=output_mol['halfedge_index'],
                        ) for t in range(len(output_mol['traj'][0]))]
                        mol_traj = []
                        for t in range(len(traj_info)):
                            try:
                                mol_traj.append(
                                    reconstruct_from_generated_with_edges(traj_info[t], False, add_edge=add_edge))
                            except MolReconsError:
                                mol_traj.append(Chem.MolFromSmiles('O'))
                        mol_info['traj'] = mol_traj

                    del mol_info['output_mol']

                    gen_list.append(mol_info)
                    mol_list.append(mol_info)
                    logger.info('Success: %s (Score: %.4f)' % (smiles, mol_info['mol_score']))

                except Exception as e:
                    logger.warning(f'Molecule processing/evaluation failed: {e}. Skipping.')
                    pool.failed.append(mol_info)
                    continue

            pool.finished.extend(gen_list)
            print_pool_status(pool, logger)

        # --- 保存输出 ---
        sdf_dir = log_dir + '/' + f'{name}_SDF'
        os.makedirs(sdf_dir, exist_ok=True)
        sorted_mol_list = sorted(mol_list, key=lambda mol: mol['mol_score'])

        with open(os.path.join(sdf_dir, 'log.txt'), 'a') as f:
            f.write('number, smiles, sa, qed, vina, toxicity, score:' + '\n')
            for i, data_finished in enumerate(sorted_mol_list):
                f.write(
                    f"{i},{data_finished['smiles']},{data_finished['sa']:.4f},"
                    f"{data_finished['qed']:.4f},{data_finished['vina_score']:.4f},"
                    f"{data_finished.get('toxicity', 'N/A'):.4f},{data_finished['mol_score']:.4f}\n"
                )

        with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
            for i, data_finished in enumerate(sorted_mol_list):
                smiles_f.write(data_finished['smiles'] + '\n')
                rdmol = data_finished['rdmol']
                try:
                    Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, '%d.sdf' % (i)))
                except:
                    continue

                if 'traj' in data_finished:
                    writer = Chem.SDWriter(os.path.join(sdf_dir, 'traj_%d.sdf' % (i)))
                    for m in data_finished['traj']:
                        try:
                            writer.write(m)
                        except:
                            writer.write(Chem.MolFromSmiles('O'))

        if config.data.dataset == 'pdbbind':
            torch.save(pool, os.path.join(log_dir, f'samples_{name}.pt'))
        elif config.data.dataset == 'crossdocked':
            name = name.replace('/', '-')
            torch.save(pool, os.path.join(log_dir, f'samples_{name}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample/sample.yml')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=0)

    parser.add_argument('--tox_script_path', type=str, default='/Data/tianjl/DiffGui_4/ens_pred_tox_1.py')
    parser.add_argument('--tox_workdir', type=str, default='/Data/tianjl/LD50/ensemble/tmp/ensemble_tox')

    args = parser.parse_args()

    main(args)