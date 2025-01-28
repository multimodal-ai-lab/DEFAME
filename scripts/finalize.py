"""Use this script to (re)compute the final experiment results (metrics, confusion matrices etc.)."""

experiment_dir = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/DEFAME/out/mocheg/pure-gpt"

if __name__ == '__main__':  # evaluation uses multiprocessing
    from defame.eval.finalize import finalize_evaluation
    finalize_evaluation(experiment_dir=experiment_dir)
