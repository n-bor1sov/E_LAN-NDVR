import os
import cv2
from tqdm import tqdm
import torch
from torchvision.transforms import v2
import torch.nn as nn
import torchvision.models as models
import torch.utils
import torch.utils.data
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
googlenet = None
base_model = None
model = None


def get_frames_emb(video):
    # Check if model is initialized
    global googlenet
    if googlenet is None:
        googlenet = models.googlenet(weights="GoogLeNet_Weights.IMAGENET1K_V1")
        googlenet.to(device)
        googlenet.eval()

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


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(5488, 2500)  # Assuming input images are 28x28
        self.fc2 = nn.Linear(2500, 1000)
        self.fc3 = nn.Linear(1000, 500)  # Output embedding of size 500

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output embedding
        x = F.normalize(x, p=2, dim=1)  # Normalize embeddings to have unit norm
        return x


class LateFusionModel(nn.Module):
    def __init__(self, base_model):
        super(LateFusionModel, self).__init__()
        self.base_model = base_model

    def forward(self, frames):
        # Process each frame independently
        batch_size, num_frames, _, _ = frames.shape
        frame_embeddings = []
        for i in range(num_frames):
            frame = frames[:, i, :, :]  # Extract each frame
            embedding = self.base_model(frame)
            frame_embeddings.append(embedding)

        # Perform late fusion by averaging the embeddings of all frames
        #fused_embedding = torch.stack(frame_embeddings, dim=1).mean(dim=1)
        # averaging
        fused_embedding = torch.mean(frame_embeddings, dim=0)
        # zero mean
        mean_value = torch.mean(fused_embedding)
        zero_mean_emb = fused_embedding - mean_value
        # l2
        fused_embedding = torch.norm(fused_embedding, p=2)
        fused_embedding = zero_mean_emb / fused_embedding

        return fused_embedding


def create_video_emb(video):
    global base_model
    global model

    # Check if models are initialized
    if base_model is None:
        base_model = EmbeddingNet()
        base_model.load_state_dict(torch.load('models/model_base.pt'))
        base_model.to(device)
        base_model.eval()

    if model is None:
        model = LateFusionModel(base_model)
        model.load_state_dict(torch.load('models/model.pt'))
        model.to(device)
        model.eval()

    # Apply aggregation and l2 norm
    video = normalize_frames(video)

    # Forward pass
    with torch.no_grad():
        video_emb = base_model(video.unsqueeze(0).to(device)).detach().cpu().numpy()[0]

    return video_emb
