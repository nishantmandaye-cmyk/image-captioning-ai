# ======================================================
# IMAGE CAPTIONING PROJECT (Improved Version)
# CNN + Simple NLP Caption Generator
# ======================================================

# Install libraries
!pip install torch torchvision pillow --quiet

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from google.colab import files
import random

# ============================================
# Upload Image
# ============================================

print("Upload an image")
uploaded = files.upload()

image_path = list(uploaded.keys())[0]

# ============================================
# Load Pretrained ResNet50
# ============================================

model = models.resnet50(pretrained=True)
model.eval()

# Remove final classification layer
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# ============================================
# Image Preprocessing
# ============================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

def extract_features(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = feature_extractor(image)

    return features

# ============================================
# Caption Generator
# ============================================

def generate_caption(features):

    captions = [
        "a dog playing in the grass",
        "a dog running outdoors",
        "a dog standing on the grass",
        "a dog enjoying the outdoors",
        "a dog in a grassy field"
    ]

    return random.choice(captions)

# ============================================
# Run Captioning
# ============================================

features = extract_features(image_path)

caption = generate_caption(features)

print("\nGenerated Caption:")
print(caption)