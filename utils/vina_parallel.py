import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def _vina_worker(args):
    mol, config, data, mode = args

    try:
        from utils.evaluation.docking_vina import VinaDockingTask

        if mode == 'pocket':
            vina_task = VinaDockingTask.from_generated_mol(
                mol, config.model.target
            )
        else:
            if config.data.dataset == 'pdbbind':
                protein_fn = os.path.join(
                    os.path.dirname(data.protein_filename),
                    os.path.basename(data.protein_filename)[:4] + '_protein.pdb'
                )
            else:
                protein_fn = os.path.join(
                    os.path.dirname(data.protein_filename),
                    os.path.basename(data.protein_filename)[:10] + '.pdb'
                )

            vina_task = VinaDockingTask.from_generated_mol(
                mol,
                os.path.join(config.data.protein_root, protein_fn)
            )

        vina_score = vina_task.run(mode='score_only', exhaustiveness=16)

        return vina_score[0]['affinity']

    except Exception:
        return None


def parallel_vina_docking(mol_list, config, data, mode, num_workers=8):
    results = [None] * len(mol_list)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _vina_worker,
                (mol_list[i], config, data, mode)
            ): i
            for i in range(len(mol_list))
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None

    return results