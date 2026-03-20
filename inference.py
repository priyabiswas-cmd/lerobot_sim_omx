import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# -----------------------------
# device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# load model
# -----------------------------
model_id = "lerobot/smolvla_base"

policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()

# -----------------------------
# create preprocessors
# -----------------------------
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

print("Model loaded")
print(policy.config.input_features)
# -----------------------------
# load dataset
# -----------------------------
dataset = LeRobotDataset("lerobot/libero")

print("Dataset loaded")

# -----------------------------
# select episode
# -----------------------------
episode_index = 0

from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
to_idx   = dataset.meta.episodes["dataset_to_index"][episode_index]

print("Episode frame range:", from_idx, "to", to_idx)

# -----------------------------
# get frame
# -----------------------------
frame_index = from_idx
frame = dict(dataset[frame_index])
# Map dataset cameras to model cameras
frame["observation.images.camera1"] = frame["observation.images.image"]
frame["observation.images.camera2"] = frame["observation.images.image2"]

# If dataset does not have third camera, duplicate one
frame["observation.images.camera3"] = frame["observation.images.image"]
# -----------------------------
# preprocess
# -----------------------------
batch = preprocess(frame)
print(policy.config.input_features)
# -----------------------------
# inference
# -----------------------------
with torch.inference_mode():
    
    pred_action = policy.select_action(batch)

    # postprocess action
    pred_action = postprocess(pred_action)

print("\nPredicted Action:")
print(pred_action)
