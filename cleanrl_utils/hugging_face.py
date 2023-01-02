import argparse
import sys
from pathlib import Path
from pprint import pformat
from typing import List

import numpy as np

HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME = "replay.mp4"
HUGGINGFACE_README_FILE_NAME = "README.md"


def push_to_hub(
    args: argparse.Namespace,
    episodic_returns: List,
    repo_id: str,
    algo_name: str,
    folder_path: str,
    video_folder_path: str = "",
    revision: str = "main",
    create_pr: bool = False,
    private: bool = False,
):
    # Step 1: lazy import and create / read a huggingface repo
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    from huggingface_hub.repocard import metadata_eval_result, metadata_save

    api = HfApi()
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
        private=private,
    )
    # parse the default entity
    entity, repo = repo_url.split("/")[-2:]
    repo_id = f"{entity}/{repo}"

    # Step 2: clean up data
    # delete previous tfevents and mp4 files
    operations = [
        CommitOperationDelete(path_in_repo=file)
        for file in api.list_repo_files(repo_id=repo_id)
        if ".tfevents" in file or file.endswith(".mp4")
    ]

    # Step 3: Generate the model card
    algorithm_variant_filename = sys.argv[0].split("/")[-1]
    model_card = f"""
# (CleanRL) **{algo_name}** Agent Playing **{args.env_id}**
This is a trained model of a {algo_name} agent playing {args.env_id}.
The model was trained by using [CleanRL](https://github.com/vwxyzjn/cleanrl) and the most up-to-date training code can be
found [here](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/{args.exp_name}.py).
## Get Started
To use this model, please install the `cleanrl` package with the following command:
```

pip install "cleanrl[{args.exp_name}]"
python -m cleanrl_utils.enjoy --exp-name {args.exp_name} --env-id {args.env_id}
```
Please refer to the [documentation](https://docs.cleanrl.dev/get-started/zoo/) for more detail.
# Hyperparameters
```python
{pformat(vars(args))}
```
"""
    # Step 1 Upload the Metadata to the Readme
    readme_path = Path(folder_path) / HUGGINGFACE_README_FILE_NAME
    readme = model_card
    metadata = {}
    metadata["tags"] = [
        args.env_id,
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "custom-implementation",
    ]
    metadata["library_name"] = "cleanrl"
    eval = metadata_eval_result(
        model_pretty_name=algo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{np.average(episodic_returns):.2f} +/- {np.std(episodic_returns):.2f}",
        dataset_pretty_name=args.env_id,
        dataset_id=args.env_id,
    )
    metadata = {**metadata, **eval}

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    metadata_save(readme_path, metadata)
    ## Add the Intended Use of the model 
    ### if model is for traed do I can be use for pretraining a RL agent with QAT and upload the int8 model somehwere in the HG Hub
    ### If the model was trained for scratch with QAT Libiraries tell them this is an INT model
    ## Add the inferecen of the model in the ReadMe
    ## Add the size of the model in the ReadMe
    ##Upload the mp4 file
    ##Upload the model 
    ##Upload the Tensorboard event to the Hugging Face Wab
    ## Share the link to the Weight and bais to the model card
    ##upload the Source Code to the Hugging Face Web
