from huggingface_hub import snapshot_download

repo_id = "robertcowher/farama-kitchen-sac-hrl-youtube"
local_download_path = "./my_kitchen_data"


path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_download_path,
)
