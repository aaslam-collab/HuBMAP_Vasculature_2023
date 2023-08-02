import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from model import UNET
import numpy as np
import argparse
from skimage.morphology import binary_dilation
import os
import tifffile as tif
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader, Dataset

import base64
from pycocotools import _mask as coco_mask
import typing as t
import zlib

IMAGE_HEIGHT = 512 
IMAGE_WIDTH = 512  
ids = []
heights = []
widths = []
all_imgs = []
prediction_strings = []
sample = None

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        return image, os.path.basename(image_path)  # Return image and filename



def encode_binary_mask(predicted_mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if predicted_mask.dtype != np.bool_:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        predicted_mask.dtype)

  predicted_mask = np.squeeze(predicted_mask)
  if len(predicted_mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        predicted_mask.shape)

  # convert input mask to expected COCO API input --
  predicted_mask_to_encode = predicted_mask.reshape(predicted_mask.shape[0], predicted_mask.shape[1], 1)
  predicted_mask_to_encode = predicted_mask_to_encode.astype(np.uint8)
  predicted_mask_to_encode = np.asfortranarray(predicted_mask_to_encode)

  # RLE encode mask --
  predicted_encoded_mask = coco_mask.encode(predicted_mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(predicted_encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str


# Set the paths
MODEL_PATH = r"C:\Users\MGS4T\Downloads\HuBMAP_New\my_checkpoint.pth.tar"
TEST_IMAGE_PATH = r"C:\Users\MGS4T\Downloads\HuBMAP_New\images\test"
OUTPUT_PATH = r"C:\Users\MGS4T\Downloads\HuBMAP_New\images\output"

def predict(model, image_tensor, filename, device):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        predicted_mask = torch.sigmoid(output).cpu().squeeze().numpy()
        predicted_mask = (predicted_mask > 0.5).astype(np.bool_)
        predicted_mask = binary_dilation(predicted_mask)

        # Visualize Mask
        plt.imshow(predicted_mask, cmap='gray')
        plt.title(f"Predicted Mask for {filename}")
        plt.show()
        
    # Encode mask for submission
    encoded = encode_binary_mask(predicted_mask)
    
    # You can use mean or max of the mask as a proxy for "score"
    # Since U-Net doesn't provide a distinct score for the mask
    score = predicted_mask.mean()
    
    pred_string = f"0 {score} {encoded.decode('utf-8')}"
        
    c, h, w = image_tensor.shape
    ids.append(os.path.splitext(filename)[0])  # appending the image filename
    heights.append(h)
    widths.append(w)
    prediction_strings.append(pred_string)

transform = A.Compose(
    [A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),])   

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNET(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()


    image_paths = [os.path.join(TEST_IMAGE_PATH, f) for f in os.listdir(TEST_IMAGE_PATH) if f.endswith(".tif")]
    dataset = CustomDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)  # Adjust batch_size based on memory availability

    for images, filenames in dataloader:
        images = images.to(device)
        outputs = model(images)    
    # Iterate over each output mask in the batch
        for idx, image_tensor in enumerate(images):
            predict(model, image_tensor, filenames[idx], device)


    # Visualize the first test image and its prediction
    array = tif.imread(r"C:\Users\MGS4T\Downloads\HuBMAP_New\images\test\72e40acccadf.tif")
    plt.imshow(array)
    plt.savefig("original_image.png")   # Save the image to a file
    plt.show()

    # Sumbission code
    submission = pd.DataFrame()
    submission['id'] = ids
    submission['height'] = heights
    submission['width'] = widths
    submission['prediction_string'] = prediction_strings
    submission = submission.set_index('id')
    submission.to_csv("submission.csv")
    print(submission.head())
