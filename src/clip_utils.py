import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from typing import List, Tuple
import torch.nn.functional as F
from torchvision import models, transforms


class Config:
    """
    Configuration class for the CLIP model.
    """
    embed_dim: int = 512  # Embedding dimension
    transformer_embed_dim: int = 768  # Transformer embedding dimension
    max_len: int = 32  # Maximum text length
    text_model: str = "distilbert-base-multilingual-cased"  # Text model name
    clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class Tokenizer:
    '''
    Tokenizer class for the CLIP model.
    '''
    def __init__(self, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x, max_length=Config.max_len, truncation=True, padding=True, return_tensors="pt"
        )


def CLIP_loss(logits: torch.Tensor, device= "cuda" if torch.cuda.is_available() else "cpu") -> torch.Tensor:
    '''
    Compute the CLIP loss from the similarity matrix.
    '''
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

def metrics(similarity: torch.Tensor):
    '''
    Compute the image and caption retrieval accuracies from the similarity matrix.
    '''
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc

class Projection(nn.Module):
    '''Projection head for the CLIP model'''
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


class VisionEncoder(nn.Module):
    '''Vision encoder for the CLIP model'''
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


class TextEncoder(nn.Module):
    '''Text encoder for the CLIP model'''
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
    

class CustomCLIPModel(nn.Module):
    '''Custom CLIP model for transfer learning'''
    def __init__(self, lr: float = 1e-4) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(Config.embed_dim)
        self.caption_encoder = TextEncoder(Config.embed_dim)
        self.tokenizer = Tokenizer(AutoTokenizer.from_pretrained(Config.text_model, use_fast=False))
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, images, text):
        '''Forward pass for the model.'''
        text = self.tokenizer(text).to(self.device)

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text["input_ids"])
        similarity = caption_embed @ image_embed.T

        loss = CLIP_loss(similarity)
        img_acc, cap_acc = metrics(similarity)
        return loss, img_acc, cap_acc
    
    def embed_image(self, image):
        '''Embed an image using the vision encoder.'''
        return self.vision_encoder(image)
    
    def embed_text(self, text):
        '''Embed a text using the text encoder.'''
        text = self.tokenizer(text).to(self.device)
        return self.caption_encoder(text["input_ids"])
    
    def get_similarity_matrix(self, images, text):
        """Compute the similarity matrix between images and captions."""
        text = self.tokenizer(text).to(self.device)

        # Get the embeddings
        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text["input_ids"])
        
        # Compute similarity matrix
        similarity = caption_embed @ image_embed.T
        
        return similarity

