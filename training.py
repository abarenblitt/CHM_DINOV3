#!/usr/bin/env python3
import os
import argparse
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
from datasets import load_dataset

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
        self.all_tied_weights_keys = {}
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
    
    print("Starting training...")
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
    
    # Save the model and stop
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(model, args.model_path)
    print(f"Training Complete. Model saved to {args.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Alaska Dead Trees Model")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--model_path", type=str, default="./output/model.pt", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    
    args = parser.parse_args()
    main(args)