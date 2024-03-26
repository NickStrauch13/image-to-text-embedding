import requests
from PIL import Image
from io import BytesIO
import os
import sqlite3
import time
import csv
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BertTokenizer
import torch.nn as nn
from torchvision import models
from typing import List, Tuple
from matplotlib import pyplot as plt
# SOURCE : https://towardsdatascience.com/clip-model-and-the-importance-of-multimodal-embeddings-1c8f6b13bf72
'''
This script downloads images from the Art Institute of Chicago's API and saves them to a directory.'''
def download_images(image_links, descriptions, base_dir="/kaggle/working/art_images"):
    
    os.makedirs(base_dir, exist_ok=True)
    metadata_path = os.path.join(base_dir, "metadata.csv")
    with open(metadata_path, 'w', newline='') as csvfile:
        metadata_writer = csv.writer(csvfile)
        metadata_writer.writerow(['filename', 'description'])
        
        for idx, (url, description) in enumerate(zip(image_links, descriptions)):
            if idx % 25 == 0:
                print(idx)
            time.sleep(0.3)
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                image_path = f"{idx}.jpg"
                full_path = os.path.join(base_dir, image_path)
                image.save(full_path)
                
                metadata_writer.writerow([image_path, description])
            except Exception as e:
                print(f"Failed to download {url}: {e}")

def download():
    connection = sqlite3.connect('artworks.db')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM artworks')

    links=[]
    descs=[]
    for row in cursor.fetchall():
        links.append(row[0])
        descs.append(row[1])

    download_images(links,descs)

# Define the ArtDataset class
# This class will load the images and their captions from the metadata CSV file.
class ArtDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        caption = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return {"image": image, "caption": caption}

'''
This function sets up the training, validation, and test datasets for the model training.
'''
def setupTrainingCSV():
    csv_file = "/kaggle/working/art_images/metadata.csv"
    img_dir = "/kaggle/working/art_images"

    # Define your transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create an instance of the ArtDataset
    # Create an instance of the ArtDataset
    art_dataset = ArtDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

    train_size = int(0.7 * len(art_dataset))  # 70% for training
    val_size = int(0.15 * len(art_dataset))  # 15% for validation
    test_size = len(art_dataset) - (train_size + val_size)  # Remaining 15% for testing

    train_dataset, val_dataset, test_dataset = random_split(art_dataset, [train_size, val_size, test_size])
    # Create the DataLoader
    clip_dataloader = DataLoader(art_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

'''
This defines the configuration class for the CLIP model training and the training function.
'''
@dataclass
class Config:
    """
    Configuration class for the CLIP training script.
    """

    embed_dim: int = 512  # Embedding dimension
    transformer_embed_dim: int = 768  # Transformer embedding dimension
    max_len: int = 32  # Maximum text length
    text_model: str = "distilbert-base-multilingual-cased"  # Text model name
    epochs: int = 5  # Number of training epochs
    batch_size: int = 128  # Batch size

'''
this function computes the CLIP loss from the similarity matrix.'''
def CLIP_loss(logits: torch.Tensor) -> torch.Tensor:
    # Assuming n is the number of classes
    n = logits.shape[1]

    # Create labels tensor
    labels = torch.arange(n).to(device)

    # Calculate cross entropy losses along axis 0 and 1
    loss_i = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
    loss_t = F.cross_entropy(logits, labels, reduction="mean")

    # Calculate the final loss
    loss = (loss_i + loss_t) / 2

    return loss
'''
this function computes the image and caption retrieval accuracies from the similarity matrix.'''
def metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc

'''
This class defines the projection head for the CLIP model.'''
class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

'''
This class defines the custom CLIP model.'''
class VisionEncoder(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        base = models.resnet34(pretrained=True)
        d_in = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.projection = Projection(d_in, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        projected_vec = self.projection(self.base(x))
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len

'''
This class defines the text encoder for the CLIP model.'''
class TextEncoder(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(Config.text_model)
        self.projection = Projection(Config.transformer_embed_dim, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(x)[0]
        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len

'''
This class defines the custom CLIP model.'''    
class Tokenizer:
    def __init__(self, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x, max_length=Config.max_len, truncation=True, padding=True, return_tensors="pt"
        )



'''
This class defines the custom CLIP model.'''
class CustomModel(nn.Module):
    def __init__(self, lr: float = 1e-4) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(Config.embed_dim)
        self.caption_encoder = TextEncoder(Config.embed_dim)
        self.tokenizer = Tokenizer(AutoTokenizer.from_pretrained(Config.text_model, use_fast=False))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, images, text):
        text = self.tokenizer(text).to(self.device)

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text["input_ids"])
        similarity = caption_embed @ image_embed.T

        loss = CLIP_loss(similarity)
        img_acc, cap_acc = metrics(similarity)
        return loss, img_acc, cap_acc
    def get_similarity_matrix(self, images, text):
        """Compute the similarity matrix between images and captions."""
        text = self.tokenizer(text).to(self.device)

        # Get the embeddings
        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text["input_ids"])
        
        # Compute similarity matrix
        similarity = caption_embed @ image_embed.T
        
        return similarity

def graph_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)  # Assumes losses were recorded after each epoch

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)  # Ensure we have a tick for every epoch
    plt.show()

'''
This function calculates the Top-K accuracy for image-caption matching.'''
def top_k_accuracy(similarity, targets, k=10):
    """Calculate Top-K accuracy for image-caption matching."""
    top_k = similarity.topk(k=k, dim=1)[1]  # Get the indices of the top k values
    correct = top_k == targets.view(-1, 1).expand_as(top_k)
    top_k_acc = correct.any(dim=1).float().mean().item()
    return top_k_acc

'''
This function evaluates the model on the test dataset.'''
def evaluate_model(model, test_loader, device, k=10):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0  # Assuming you might still want to track loss
    total_top_k_img_acc = 0
    total_top_k_cap_acc = 0
    
    with torch.no_grad():  # No need to track gradients for evaluation
        for batch in test_loader:
            images = batch["image"].to(device)
            captions = batch["caption"]
            # Assuming 'get_similarity_matrix' is a method that returns the similarity matrix
            similarity_matrix = model.get_similarity_matrix(images, captions)  # Adjusted line
            targets = torch.arange(len(similarity_matrix)).to(device)
            
            top_k_img_acc = top_k_accuracy(similarity_matrix, targets, k)
            top_k_cap_acc = top_k_accuracy(similarity_matrix.T, targets, k)  # Adjusted for caption to image
            
            # Assuming loss calculation is separate or not needed for this part
            # total_loss += loss.item()  # You would calculate loss elsewhere if needed
            
            # Accumulate Top-K accuracies
            total_top_k_img_acc += top_k_img_acc
            total_top_k_cap_acc += top_k_cap_acc
    
    # Calculate average metrics
    avg_top_k_img_acc = total_top_k_img_acc / len(test_loader)
    avg_top_k_cap_acc = total_top_k_cap_acc / len(test_loader)
    
    print(f"Test Top-{k} Image Accuracy: {avg_top_k_img_acc:.4f}, Test Top-{k} Caption Accuracy: {avg_top_k_cap_acc:.4f}")
    return avg_top_k_img_acc, avg_top_k_cap_acc

def train_model(train_loader, val_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CustomModel().to(device)
    # Define optimizer
    optimizer = torch.optim.Adam([
        {'params': model.vision_encoder.parameters()},
        {'params': model.caption_encoder.parameters()}
    ], lr=model.lr)

    #TESTING WITH VALIDATION
    start_epoch = 0
    num_epochs = 20
    train_losses=[]
    val_losses=[]
    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set the model to training mode
        train_loss, train_img_acc, train_cap_acc = 0.0, 0.0, 0.0
        count = 0
        
        # Training Phase
        for batch in train_loader:
            count += 1
            image = batch["image"].to(device)
            text = batch["caption"]
            optimizer.zero_grad()  # Zero the gradients
            loss, img_acc, cap_acc = model(image, text)  # Forward pass
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            # Accumulate metrics
            train_loss += loss.item()
            train_img_acc += img_acc.item()
            train_cap_acc += cap_acc.item()
            
            if count % 100 == 0:
                print(f"Batch {count}, Training Loss: {loss.item()}")
            
        avg_train_loss = train_loss / count
        avg_train_img_acc = train_img_acc / count
        avg_train_cap_acc = train_cap_acc / count
        print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss}, Image Acc: {avg_train_img_acc}, Caption Acc: {avg_train_cap_acc}")
        train_losses.append(avg_train_loss)
        # Validation Phase
        model.eval()  # Set the model to evaluation mode
        val_loss, val_img_acc, val_cap_acc = 0.0, 0.0, 0.0
        with torch.no_grad():  # No need to track gradients
            for batch in val_loader:
                image = batch["image"].to(device)
                text = batch["caption"]
                loss, img_acc, cap_acc = model(image, text)
                
                # Accumulate metrics
                val_loss += loss.item()
                val_img_acc += img_acc.item()
                val_cap_acc += cap_acc.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_img_acc = val_img_acc / len(val_loader)
        avg_val_cap_acc = val_cap_acc / len(val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}, Image Acc: {avg_val_img_acc}, Caption Acc: {avg_val_cap_acc}")
        val_losses.append(avg_val_loss)
    # save the model weights
    torch.save(model.state_dict(), "art_clip_model.pth")
    return train_losses, val_losses, model, device

'''
This function will download the images from the Art Institute of Chicago's API and train the CLIP model with the downloaded images and their descriptions.'''
def main():
    download()
    train_loader, val_loader, test_loader = setupTrainingCSV()
    train_losses, val_losses,model, device= train_model(train_loader, val_loader)
    graph_losses(train_losses, val_losses)
    for k in [1,3, 5, 10,20]:
        evaluate_model(model, test_loader, device, k=k)

    
