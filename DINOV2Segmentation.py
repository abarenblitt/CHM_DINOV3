#!/usr/bin/env python3
import os
import argparse
import warnings
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors, transforms
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
from datasets import load_dataset
import evaluate

# Ignore spammy warnings during the loop
warnings.filterwarnings("ignore")

# --- MODEL DEFINITIONS ---
class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()
        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)
        return self.classifier(embeddings)

class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    _tied_weights_keys = []
    def __init__(self, config):
        super().__init__(config)
        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(config.hidden_size, 32, 32, config.num_labels)

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
        outputs = self.dinov2(pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        patch_embeddings = outputs.last_hidden_state[:,1:,:]
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:
            weights = torch.tensor([1.0, 10.0]).to(pixel_values.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights) 
            labels_squeezed = labels.squeeze(1) if labels.dim() == 4 else labels
            loss = loss_fct(logits, labels_squeezed)

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# --- DATASET DEFINITION ---
class SegmentationDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = tv_tensors.Image(item["image"]) 
        mask = tv_tensors.Mask(item["label"])
        transformed_image, transformed_mask = self.transform(image, mask)
        target = transformed_mask.to(torch.long)
        return transformed_image, target

def collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    return batch

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    id2label = {0: "live", 1: "dead"}
    
    # ---------------------------------------------------------
    # 1. TRAIN MODEL (Skipped if model already exists)
    # ---------------------------------------------------------
    if not os.path.exists(args.model_path):
        print("Model not found at specified path. Starting training...")
        dataset = load_dataset('saking3/alaska_dead_trees', token=args.hf_token)
        
        ADE_MEAN = [123.675 / 255, 116.280 / 255, 103.530 / 255]
        ADE_STD = [58.395 / 255, 57.120 / 255, 57.375 / 255]
        
        train_transform = v2.Compose([
            v2.Resize((448,448), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ])
        
        train_dataset = SegmentationDataset(dataset["train"], transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)
        
        model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", id2label=id2label, num_labels=len(id2label))
        
        for name, param in model.named_parameters():
            if name.startswith("dinov2"):
                param.requires_grad = False
                
        model.to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        for epoch in range(args.epochs):
            print(f"Epoch: {epoch+1}/{args.epochs}")
            for idx, batch in enumerate(train_dataloader):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if idx % 100 == 0:
                    print(f"Step {idx} - Loss: {loss.item():.4f}")
        
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        torch.save(model, args.model_path)
        print(f"Model saved to {args.model_path}")
    else:
        print(f"Loading pre-trained model from {args.model_path}")
        model = torch.load(args.model_path, map_location=device, weights_only=False)
        model.to(device)

    # ---------------------------------------------------------
    # 2. INFERENCE ON G-LiHT IMAGE
    # ---------------------------------------------------------
    model.eval()
    vsicurl_path = f"/vsicurl/{args.tif_url}"
    
    print(f"Opening raster: {vsicurl_path}")
    with rasterio.open(vsicurl_path) as src:
        image_data = src.read()
        profile = src.profile  # Store spatial metadata for the output

    image_rgb = image_data[:3, :, :]
    _, height, width = image_rgb.shape

    infer_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32), 
        transforms.Resize((448, 448), antialias=True), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    PATCH_SIZE = 256  
    output_map = np.zeros((height, width), dtype=np.uint8)

    print(f"Starting inference on {width}x{height} image...")
    
    with torch.no_grad(): 
        for y in range(0, height, PATCH_SIZE):
            for x in range(0, width, PATCH_SIZE):
                y_end = min(y + PATCH_SIZE, height)
                x_end = min(x + PATCH_SIZE, width)
                
                patch = image_rgb[:, y:y_end, x:x_end]
                patch_tensor = torch.from_numpy(patch)
                
                pad_h = PATCH_SIZE - patch_tensor.shape[1]
                pad_w = PATCH_SIZE - patch_tensor.shape[2]
                
                if pad_h > 0 or pad_w > 0:
                    patch_tensor = F.pad(patch_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
                
                input_tensor = infer_transform(patch_tensor).unsqueeze(0).to(device)
                prediction = model(input_tensor)
                logits = prediction.logits

                probs = F.softmax(logits, dim=1)
                tree_probs = probs[:, 1, :, :]
                predicted_class = (tree_probs > args.threshold).long()
                
                predicted_class_256 = F.interpolate(
                    predicted_class.unsqueeze(1).float(), 
                    size=(PATCH_SIZE, PATCH_SIZE),
                    mode='nearest'
                ).squeeze()
                
                pred_np = predicted_class_256.cpu().numpy().astype(np.uint8)
                
                valid_h = y_end - y
                valid_w = x_end - x
                output_map[y:y_end, x:x_end] = pred_np[:valid_h, :valid_w]
                
            if y % (PATCH_SIZE * 10) == 0:
                print(f"Processed row {y} of {height}")

    # ---------------------------------------------------------
    # 3. SAVE OUTPUT AS GEOTIFF
    # ---------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "dead_trees_prediction.tif")
    
    # Update profile to match 1-band 8-bit output
    profile.update(
        count=1,
        dtype=rasterio.uint8,
        compress='lzw'
    )
    
    print(f"Writing results to {out_path}...")
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(output_map, 1)

    print("DPS Job Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alaska Dead Trees DPS Algorithm")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--tif_url", type=str, required=True, help="URL to the G-LiHT image")
    parser.add_argument("--model_path", type=str, default="./model/model.pt", help="Path to save/load model")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory for DPS outputs")
    parser.add_argument("--threshold", type=float, default=0.30, help="Classification probability threshold")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (if model not found)")
    
    args = parser.parse_args()
    main(args)