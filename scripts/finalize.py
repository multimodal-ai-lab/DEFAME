"""Use this script to (re)compute the final experiment results (metrics, confusion matrices etc.)."""

experiment_dir = "/path/to/exp/dir"

if __name__ == '__main__':  # evaluation uses multiprocessing
    from defame.eval.finalize import finalize_evaluation
    finalize_evaluation(experiment_dir=experiment_dir)
