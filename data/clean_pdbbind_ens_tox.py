import os
import sys
import pickle
import argparse
import tempfile
import subprocess
import shlex
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

KMAP = {'Ki': 1, 'Kd': 2, 'IC50': 3}


class EnsembleToxicityPredictor:
    """
    调用 ens_pred_tox.py 的集成毒性预测器
    输入一批 smiles，输出对应的 pred_ensemble
    """

    def __init__(self, ensemble_script, workdir=None, python_exec=None):
        self.ensemble_script = ensemble_script
        self.workdir = workdir or tempfile.mkdtemp(prefix="ensemble_tox_")
        self.python_exec = python_exec or sys.executable

        os.makedirs(self.workdir, exist_ok=True)

    def predict_from_smiles(self, smiles_list):
        """
        批量预测 smiles 列表的毒性
        返回长度与输入一致的列表，失败项记为 None
        """
        input_txt = os.path.join(self.workdir, "tox_input.txt")
        output_csv = os.path.join(self.workdir, "tox_output.csv")

        with open(input_txt, "w", encoding="utf-8") as f:
            for smi in smiles_list:
                f.write(str(smi).strip() + "\n")

        cmd = [
            self.python_exec,
            self.ensemble_script,
            "--txt", input_txt,
            "--output", output_csv,
        ]

        print("\n[调用集成毒性预测脚本]")
        print(" ".join(shlex.quote(str(x)) for x in cmd))

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("集成毒性预测失败:")
            print(e.stdout)
            return [None] * len(smiles_list)

        if not os.path.exists(output_csv):
            print(f"未找到集成预测输出文件: {output_csv}")
            return [None] * len(smiles_list)

        try:
            pred_df = pd.read_csv(output_csv)
        except Exception as e:
            print(f"读取集成预测结果失败: {e}")
            return [None] * len(smiles_list)

        if "pred_ensemble" not in pred_df.columns:
            print("输出文件中缺少 pred_ensemble 列")
            return [None] * len(smiles_list)

        preds = pd.to_numeric(pred_df["pred_ensemble"], errors="coerce").tolist()

        if len(preds) != len(smiles_list):
            print(
                f"警告: 集成预测结果数量与输入不一致, "
                f"input={len(smiles_list)}, output={len(preds)}"
            )
            if len(preds) < len(smiles_list):
                preds = preds + [None] * (len(smiles_list) - len(preds))
            else:
                preds = preds[:len(smiles_list)]

        return preds

    def predict_from_mols(self, mols, timeout=3600, batch_size=64):
        """
        与原 ToxicityPredictor 接口兼容
        """
        smiles_list = []
        for mol in mols:
            if mol is None:
                smiles_list.append(None)
            else:
                try:
                    smiles_list.append(Chem.MolToSmiles(mol))
                except Exception:
                    smiles_list.append(None)

        valid_smiles = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            if smi is not None:
                valid_smiles.append(smi)
                valid_indices.append(i)

        final_preds = [None] * len(mols)

        if len(valid_smiles) == 0:
            return final_preds

        pred_vals = self.predict_from_smiles(valid_smiles)

        for idx, pred in zip(valid_indices, pred_vals):
            final_preds[idx] = pred

        return final_preds


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
        print("✓ 毒性预测完成")
        return toxicities
    except Exception as e:
        print(f"批量毒性预测失败: {e}")
        return [None] * len(mols)


def main(args):
    # 初始化毒性预测器
    toxicity_predictor = None
    if args.ensemble_script:
        try:
            print("=" * 60)
            print("初始化集成毒性预测器...")
            print(f"集成脚本路径: {args.ensemble_script}")
            print(f"调用Python: {args.ensemble_python}")
            toxicity_predictor = EnsembleToxicityPredictor(
                ensemble_script=args.ensemble_script,
                workdir=args.ensemble_workdir,
                python_exec=args.ensemble_python
            )
            print("✓ 集成毒性预测器初始化成功")
            print("=" * 60)
        except Exception as e:
            print(f"✗ 无法初始化集成毒性预测器: {e}")
            print("将不进行毒性预测")
            print("=" * 60)
    else:
        print("=" * 60)
        print("警告: 未提供集成毒性预测脚本")
        print("将不进行毒性预测")
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

                if not os.path.exists(ligand_path):
                    continue

                if os.path.getsize(ligand_path) == 0:
                    mol2_path = ligand_path.replace('.sdf', '.mol2')
                    if os.path.exists(mol2_path):
                        os.system(f'obabel {mol2_path} -O {ligand_path}')

                mol = Chem.SDMolSupplier(ligand_path)[0]
                if mol is None:
                    print(f"无法加载分子: {pdbid}")
                    continue

                logp = round(Descriptors.MolLogP(mol), 4)
                tpsa = round(rdMolDescriptors.CalcTPSA(mol), 4)
                sascore = round(sascorer.calculateScore(mol), 4)
                qed_score = round(QED.qed(mol), 4)
                smiles = Chem.MolToSmiles(mol)

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
                round(float(toxicity), 4)
            ))
        else:
            num_discarded += 1

    if num_discarded > 0:
        print(f"因毒性预测失败，共舍弃 {num_discarded} 个条目。")

    new_index_path = os.path.join(args.source, "index_ensemble.pkl")
    with open(new_index_path, "wb") as f:
        pickle.dump(index, f)

    print(f"\n✓ 处理完成!")
    print(f"  - 处理了 {len(index)} 个蛋白-配体对")
    print(f"  - 索引文件保存到: {new_index_path}")

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
        "--ensemble-script",
        type=str,
        default="/Data/tianjl/DiffGui_4/ens_pred_tox_1.py",
        help="集成毒性预测脚本路径"
    )
    parser.add_argument(
        "--ensemble-python",
        type=str,
        default="/Data/tianjl/anaconda3/envs/diffgui/bin/python",
        help="运行集成毒性预测脚本的 Python 解释器"
    )
    parser.add_argument(
        "--ensemble-workdir",
        type=str,
        default="/Data/tianjl/LD50/ensemble/workdir_pdbbind",
        help="集成毒性预测中间目录"
    )
    args = parser.parse_args()

    main(args)