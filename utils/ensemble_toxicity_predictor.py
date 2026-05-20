import os
import pandas as pd
import subprocess
import uuid


class EnsembleToxicityPredictor:
    def __init__(self, script_path, workdir):
        self.script_path = script_path

        self.workdir = os.path.join(workdir, uuid.uuid4().hex)

        os.makedirs(self.workdir, exist_ok=True)

    def predict_batch(self, smiles_list):
        try:
            input_txt = os.path.join(self.workdir, "batch_input.txt")
            output_csv = os.path.join(self.workdir, "batch_output.csv")

            with open(input_txt, "w") as f:
                for smi in smiles_list:
                    f.write(smi + "\n")

            cmd = [
                "/Data/tianjl/anaconda3/envs/deepblock/bin/python",
                self.script_path,
                "--txt", input_txt,
                "--output", output_csv,
                "--workdir", self.workdir
            ]

            subprocess.run(cmd, check=True)

            df = pd.read_csv(output_csv)
            return df["pred_ensemble"].tolist()

        except Exception as e:
            print(f"[Batch Toxicity Error] {e}")
            return [None] * len(smiles_list)