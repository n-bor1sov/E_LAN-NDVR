import os
import cv2
from tqdm import tqdm
import torch
from torchvision.transforms import v2
import torch.nn as nn

import torch.utils
import torch.utils.data
from torch.nn import functional as F


def get_frames_emb(video):
    # Dictionary to store the maxpooled outputs for each Conv2d layer
    maxpooled_outputs = {}

    # Hook function to apply global max pooling
    def maxpool_hook(module, input, output):
        maxpooled, _ = torch.max(output, dim=2)     # Max-pooling over height (H)
        maxpooled, _ = torch.max(maxpooled, dim=2)  # Max-pooling over width (W)
        maxpooled_outputs[module] = maxpooled

    # Register hooks for each Conv2d layer in AlexNet
    def hook_model(googlenet):
        for name, layer in googlenet.named_children():
            if "inception" in name:
                layer.register_forward_hook(maxpool_hook)

    # Concatenate tensors function
    def get_maxpooled_frames(maxpooled_outputs):
        # Get all the maxpooled tensors from the dictionary
        all_tensors = list(maxpooled_outputs.values())
        # Concatenate them along the second dimension (channels)
        concatenated_tensor = torch.cat(all_tensors, dim=1)
        return concatenated_tensor

    # Register hooks
    hook_model(googlenet)

    with torch.no_grad():
        googlenet(video.to(device))

        # Get maxpooled frames
        result = get_maxpooled_frames(maxpooled_outputs).cpu().detach()

        # Clear maxpooled_outputs for the next batch
        maxpooled_outputs.clear()

        return result


def create_per_frame_embeddings(path):
    # Frame transformations
    transforms = v2.Compose([
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Read video
    counter = 0
    frames = []
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret and counter % (int(fps) // 2) == 0:
            img = torch.from_numpy(img).permute(2, 0, 1)/255
            frames.append(transforms(img))
        counter = (counter + 1) % (int(fps) // 2)
    video = torch.stack(frames) # dimensions (T, C, H, W)

    # Create emb by maxpool from alexnet
    video = get_frames_emb(video)

    return video


def normalize_frames(video):
    # Step 1: Average across frames (mean along the first dimension)
    avg_emb = torch.mean(video, dim=0)

    # Step 2: Zero-mean normalization (subtract the mean of the vector)
    mean_value = torch.mean(avg_emb)
    zero_mean_emb = avg_emb - mean_value

    # Step 3: ℓ2-normalization (normalize by the L2 norm)
    l2_norm = torch.norm(zero_mean_emb, p=2)
    l2_normalized_emb = zero_mean_emb / l2_norm

    return l2_normalized_emb


