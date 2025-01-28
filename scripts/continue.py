"""Use this script to resume an incomplete evaluation run."""

experiment_dir = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/DEFAME/out/mocheg/2024-11-10_23-11-ACC-60-f1-54"

if __name__ == '__main__':  # evaluation uses multiprocessing
    from defame.eval.continute import continue_evaluation
    continue_evaluation(experiment_dir=experiment_dir)
