from huggingface_hub import Repository, scan_cache_dir

def manage_hf_models():
    """
    Lists and deletes selected models from the Hugging Face cache.

    Args:
        cache_dir (str): Path to the Hugging Face cache directory. Default is "~/.cache/huggingface/hub".

    Returns:
        None
    """
    cache = scan_cache_dir()
    bytes_per_mb = 1024 ** 2
    hashes = dict()
    # List all models in the cache
    print("Listing models in the cache:")
    for repo in cache.repos:
        print(f"Repo ID: {repo.repo_id}")
        print(f"Repo Size: {int(repo.size_on_disk)//bytes_per_mb} MB")
        hashes[repo.repo_id] = []
        for revision in repo.revisions:
            hashes[repo.repo_id].append(revision.commit_hash)

    # Prompt the user to enter the names of the models to delete
    models_to_delete = input("\nEnter the names of the models you wish to delete, separated by commas. Press X to opt out:\n").split(',')

    # Delete the selected models
    for model_name in models_to_delete:
        if model_name == "X" or model_name == "x":
            return
        try:
            for hash in hashes[model_name]:
                delete_strategy = cache.delete_revisions(hash)
                print(f"Will free {delete_strategy.expected_freed_size_str}.")
                delete_strategy.execute()
            print(f"Deleted {model_name}.")
        except Exception as e:
            print(f"Failed to delete {model_name}: {e}")

if __name__ == "__main__":
    manage_hf_models()