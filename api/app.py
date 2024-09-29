from flask import Flask, request, jsonify
from flask import abort
from uuid import UUID, uuid4
from typing import Dict
import requests
import os
import torch
from torch import nn
import cv2
import torchvision.models as models
import cv2


app = Flask(__name__)

# Define the video link request and response data models
class VideoLinkRequest:
    def __init__(self, link: str):
        self.link = link

class VideoLinkResponse:
    def __init__(self, is_duplicate: bool, duplicate_for: str = None):
        self.is_duplicate = is_duplicate
        self.duplicate_for = duplicate_for

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

# Define a dictionary to store video duplicates (in a real-world scenario, this would be a database)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

googlenet = models.googlenet(weights="GoogLeNet_Weights.IMAGENET1K_V1")
googlenet.to(device)
googlenet.eval()

base_model = EmbeddingNet()
base_model.load_state_dict(torch.load('../models/model_base.pt'))
base_model.to(device)
base_model.eval()

model = LateFusionModel(base_model)
model.load_state_dict(torch.load('../models/model.pt'))
model.to(device)
model.eval()

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


def create_video_emb(video):
    global base_model
    global model

    # Apply aggregation and l2 norm
    video = normalize_frames(video)

    # Forward pass
    with torch.no_grad():
        video_emb = base_model(video.unsqueeze(0).to(device)).detach().cpu().numpy()[0]

    return video_emb

def request_to_model():
    pass

def download_video(url, filename):
    try:
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            # Get the total size of the file
            total_size = int(response.headers.get('content-length', 0))
            
            # Create a progress bar
            print(f"Downloading {filename} ({total_size / (1024 * 1024):.2f} MB)...")

            filename = f'../data/videos/{filename}'
            # Create a directory if it doesn't exist
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Open the file in binary write mode
            with open(filename, 'wb') as f:
                # Write the file in chunks to avoid loading the entire file into memory
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

            print("Download complete!")
        else:
            print("Failed to download file")

    except Exception as e:
        print(f"Error downloading file: {e}")


@app.route('/check-video-duplicate', methods=['POST'])
def check_video_duplicate():
    try:
        data = request.get_json()
        if 'link' not in data:
            return jsonify({'error': 'Missing link parameter'}), 400

        link = data['link']
        video_id = UUID(link.split('/')[-1])  # Extract UUID from the link

        download_video(link, str(video_id))
        
        concatenated_frames = get_frames_emb(str(video_id))
        
        print(concatenated_frames.shape)
        
        # Скрипт для веторизации видео
        
        # Скрипт для эмбединга видео
        
        # Скрипт для запроса к Qdrant
        
        # Simulate a duplicate check (in a real-world scenario, this would be a database query)
        # if video_id in video_duplicates:
        #     return jsonify(VideoLinkResponse(is_duplicate=True, duplicate_for=str(video_id)).to_dict()), 200
        # else:
        #     video_duplicates[video_id] = True
        #     return jsonify(VideoLinkResponse(is_duplicate=False).to_dict()), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Convert data models to dictionaries
def to_dict(obj):
    if isinstance(obj, VideoLinkRequest):
        return {'link': obj.link}
    elif isinstance(obj, VideoLinkResponse):
        return {'is_duplicate': obj.is_duplicate, 'duplicate_for': obj.duplicate_for}

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="3010")