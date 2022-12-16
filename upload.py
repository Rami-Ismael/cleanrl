import os
from dotenv import load_dotenv

from huggingface_hub import HfApi

load_dotenv()

HF_KEY = os.getenv("HF_KEY")

## Create a new repo on Hugging Face Hub

api = HfApi()
api.create_repo(token=HF_KEY, 
                repo_id="Rami/test-practice",
                exist_ok=True,
                repo_type="model",
)

## Upload your latest to your run

api.upload_file(
    path_or_fileobj="runs/CartPole-v1__functional_dqn__42__1665377997/events.out.tfevents.1665378005.localhost-live.attlocal.net.70070.0",
    path_in_repo= "runs/CartPole-v1__functional_dqn__42__1665377997/events.out.tfevents.1665378005.localhost-live.attlocal.net.70070.0",
    repo_id="Rami/test-practice",
    repo_type="model",
    token = HF_KEY,
)
## Upload your model to the Hugging Face Hub

api.upload_file(
    path_or_fileobj="q_network.pt",
    path_in_repo= "pytorch_model.bin",
    repo_id="Rami/test-practice",
    repo_type="model",
    token=HF_KEY,
)

