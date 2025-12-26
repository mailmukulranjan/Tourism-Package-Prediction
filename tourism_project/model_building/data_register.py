from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# The Hugging Face token is retrieved from the environment variable 'HF_TOKEN'.
# Please ensure this environment variable is set before running this cell.
# In Google Colab, you can set it using:
# import os
# os.environ["HF_TOKEN"] = "YOUR_HUGGING_FACE_TOKEN"
# Or by using Colab secrets for better security (Recommended).
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    # Changed the error message to be more precise about the expected environment variable.
    raise RuntimeError("HF token not found. Please set the 'HF_TOKEN' environment variable.")

api = HfApi(token=HF_TOKEN)

repo_id = "mailmukulranjan/tourism-package-prediction"
repo_type = "dataset"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
