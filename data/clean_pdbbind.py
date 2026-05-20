import os
import sys
import pickle
import argparse
import tempfile

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

sys.path.append('..')
from utils.toxicity_predictor import ToxicityPredictor


KMAP = {'Ki': 1, 'Kd': 2, 'IC50': 3}


def batch_predict_toxicity(mols, smiles_list, toxicity_predictor):
    """
    批量预测毒性

    Args:
        mols: RDKit Mol 对象列表
        smiles_list: SMILES 字符串列表
        toxicity_predictor: 毒性预测器实例

    Returns:
        毒性值列表
    """
    if toxicity_predictor is None:
        print("警告: 未提供毒性预测器,所有条目将被舍弃")
        return [None] * len(mols)

    print(f"批量预测 {len(mols)} 个分子的毒性...")
    try:
        toxicities = toxicity_predictor.predict_from_mols(
            mols,
            timeout=3600,
            batch_size=64
        )
        print(f"✓ 毒性预测完成")
        return toxicities
    except Exception as e:
        print(f"批量毒性预测失败: {e}")
        return [None] * len(mols)


def main(args):
    # 初始化毒性预测器
    toxicity_predictor = None
    if args.toxicity_model and args.chemprop_python:
        try:
            print("=" * 60)
            print("初始化毒性预测器...")
            print(f"模型路径: {args.toxicity_model}")
            print(f"ChemProp Python: {args.chemprop_python}")
            toxicity_predictor = ToxicityPredictor(
                model_path=args.toxicity_model,
                chemprop_python=args.chemprop_python
            )
            print("✓ 毒性预测器初始化成功")
            print("=" * 60)
        except Exception as e:
            print(f"✗ 无法初始化毒性预测器: {e}")
            print("将使用简单规则估算毒性")
            print("=" * 60)
    else:
        print("=" * 60)
        print("警告: 未提供毒性模型参数")
        print("将使用简单规则估算毒性")
        print("=" * 60)

    # 第一遍: 收集所有有效的分子
    print("\n第一步: 加载数据并计算基础属性...")
    index_temp = []
    mols = []
    smiles_list = []

    index_path = os.path.join(args.source, "index/INDEX_general_PL_data.2020")
    with open(index_path, 'r') as fr:
        lines = fr.readlines()

    for line in tqdm(lines, desc="处理分子"):
        if line.startswith('#'):
            continue
        else:
            try:
                pdbid, res, year, pka, kv = line.split('//')[0].strip().split()
                kind = [v for k, v in KMAP.items() if k in kv]
                assert len(kind) == 1

                protein_fn = pdbid + "/" + pdbid + "_protein.pdb"
                ligand_fn = pdbid + "/" + pdbid + "_ligand.sdf"
                ligand_path = os.path.join(args.source, ligand_fn)

                # 检查文件是否存在
                if not os.path.exists(ligand_path):
                    continue

                # 如果文件为空,尝试转换
                if os.path.getsize(ligand_path) == 0:
                    mol2_path = ligand_path.replace('.sdf', '.mol2')
                    if os.path.exists(mol2_path):
                        os.system(f'obabel {mol2_path} -O {ligand_path}')

                # 加载分子
                mol = Chem.SDMolSupplier(ligand_path)[0]
                if mol is None:
                    print(f"无法加载分子: {pdbid}")
                    continue

                # 计算基础属性
                logp = round(Descriptors.MolLogP(mol), 4)
                tpsa = round(rdMolDescriptors.CalcTPSA(mol), 4)
                sascore = round(sascorer.calculateScore(mol), 4)
                qed_score = round(QED.qed(mol), 4)

                # 获取 SMILES
                smiles = Chem.MolToSmiles(mol)

                # 保存临时数据
                index_temp.append({
                    'pdbid': pdbid,
                    'protein_fn': protein_fn,
                    'ligand_fn': ligand_fn,
                    'logp': logp,
                    'tpsa': tpsa,
                    'sascore': sascore,
                    'qed_score': qed_score,
                    'pka': pka,
                    'kind': kind[0],
                    'smiles': smiles
                })
                mols.append(mol)
                smiles_list.append(smiles)

            except Exception as e:
                print(f"处理 {pdbid} 失败: {str(e)}")
                continue

    print(f"✓ 成功处理 {len(index_temp)} 个分子")

    # 第二步: 批量预测毒性
    print("\n第二步: 批量预测毒性...")
    toxicities = batch_predict_toxicity(mols, smiles_list, toxicity_predictor)


    # 第三步: 组装最终索引
    print("\n第三步: 组装最终索引...")
    index = []
    num_discarded = 0
    for entry, toxicity in zip(index_temp, toxicities):
        if toxicity is not None:
            # 格式: (protein_fn, ligand_fn, pdbid, logp, tpsa, sascore, qed_score, pka, kind, toxicity)
            index.append((
                entry['protein_fn'],
                entry['ligand_fn'],
                entry['pdbid'],
                entry['logp'],
                entry['tpsa'],
                entry['sascore'],
                entry['qed_score'],
                entry['pka'],
                entry['kind'],
                round(toxicity, 4)
            ))
        else:
            num_discarded += 1

    if num_discarded > 0:
        print(f"因毒性预测失败，共舍弃 {num_discarded} 个条目。")

    # 保存索引
    new_index_path = os.path.join(args.source, "index.pkl")
    with open(new_index_path, "wb") as f:
        pickle.dump(index, f)

    print(f"\n✓ 处理完成!")
    print(f"  - 处理了 {len(index)} 个蛋白-配体对")
    print(f"  - 索引文件保存到: {new_index_path}")

    # 打印统计信息
    print("\n" + "=" * 60)
    print("属性统计信息:")
    print("=" * 60)

    import numpy as np

    logps = [item[3] for item in index]
    tpsas = [item[4] for item in index]
    sascores = [item[5] for item in index]
    qeds = [item[6] for item in index]
    pkas = [float(item[7]) for item in index]
    toxs = [item[9] for item in index]

    def print_stats(name, values):
        print(f"\n{name}:")
        print(f"  Mean:   {np.mean(values):.4f}")
        print(f"  Median: {np.median(values):.4f}")
        print(f"  Min:    {np.min(values):.4f}")
        print(f"  Max:    {np.max(values):.4f}")
        print(f"  Std:    {np.std(values):.4f}")

    print_stats("LogP", logps)
    print_stats("TPSA", tpsas)
    print_stats("SA Score", sascores)
    print_stats("QED", qeds)
    print_stats("pKa (Affinity)", pkas)
    print_stats("Toxicity", toxs)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='清理 PDBbind 数据集并计算分子属性(包括毒性)'
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./PDBbind_v2020",
        help="PDBbind 数据源路径"
    )
    parser.add_argument(
        "--toxicity-model",
        type=str,
        default="/Data/tianjl/TOX/drug_LD50_regression_model/drug_LD50_v5_all_optimized/model_0/best.pt",
        help="ChemProp 毒性模型路径"
    )
    parser.add_argument(
        "--chemprop-python",
        type=str,
        default="/Data/tianjl/anaconda3/envs/chemprop/bin/python",
        help="ChemProp 环境的 Python 解释器路径"
    )
    args = parser.parse_args()

    main(args)
