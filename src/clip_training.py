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
from clip_utils import Config, Tokenizer, CLIP_loss, metrics, CustomCLIPModel, VisionEncoder, TextEncoder, Projection

# SOURCE : https://towardsdatascience.com/clip-model-and-the-importance-of-multimodal-embeddings-1c8f6b13bf72
'''
This script downloads images from the Art Institute of Chicago's API and saves them to a directory.'''
def download_images(image_links, descriptions, base_dir="/kaggle/working/art_images"):
    # Create the directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    metadata_path = os.path.join(base_dir, "metadata.csv")
    # Download the images and save them to the directory
    with open(metadata_path, 'w', newline='') as csvfile:
        metadata_writer = csv.writer(csvfile)
        metadata_writer.writerow(['filename', 'description'])
        # Loop through the image links and download each image
        for idx, (url, description) in enumerate(zip(image_links, descriptions)):
            if idx % 25 == 0:
                print(idx)
            time.sleep(0.3)
            # Download the image
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                image_path = f"{idx}.jpg"
                full_path = os.path.join(base_dir, image_path)
                image.save(full_path)
                
                metadata_writer.writerow([image_path, description])
            except Exception as e:
                print(f"Failed to download {url}: {e}")

'''
this function downloads the images from the Art Institute of Chicago's API and saves them to a directory.'''
def download(base_dir="/kaggle/working/art_images"):
    connection = sqlite3.connect('../notebooks/artworks.db')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM artworks')

    links=[]
    descs=[]
    for row in cursor.fetchall():
        links.append(row[0])
        descs.append(row[1])

    download_images(links,descs, base_dir=base_dir)

# Define the ArtDataset class
# This class will load the images and their captions from the metadata CSV file.
class ArtDataset(Dataset):
    '''
    This class loads the images and their captions from the metadata CSV file.'''
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
    '''
    This function returns the length of the dataset.'''
    def __len__(self):
        return len(self.img_labels)
    '''
    This function returns a sample from the dataset.'''
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
def setupTrainingCSV(csv_file="/kaggle/working/art_images/metadata.csv", img_dir="/kaggle/working/art_images"):

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
This function graphs the training and validation losses over the epochs.'''
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
    # Evaluate the model
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

'''
This function trains the CLIP model on the training dataset and evaluates it on the validation dataset.'''
def train_model(train_loader, val_loader):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CustomCLIPModel().to(device)
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
            # Zero the gradients
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
        # Calculate average metrics
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
        # Calculate average metrics
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
    # Download the images
    download(base_dir='../data/art_images')
    # Set up the training, validation, and test datasets 
    # May need to tweak the paths if this is run in a different environment than Kaggle
    train_loader, val_loader, test_loader = setupTrainingCSV()
    # Train the model
    train_losses, val_losses,model, device= train_model(train_loader, val_loader)
    # Graph the losses
    graph_losses(train_losses, val_losses)
    # Evaluate the model on the test dataset
    for k in [1,3, 5, 10,20]:
        evaluate_model(model, test_loader, device, k=k)

'''
This script downloads images from the Art Institute of Chicago's API and saves them to a directory.'''
if __name__ == '__main__':
    main()