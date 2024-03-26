from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
from torch import Tensor
import torch.nn as nn
from datasets import load_dataset
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomTextEmbeddingModel(torch.nn.Module):
    def __init__(self, original_model, output_dim):
        super(CustomTextEmbeddingModel, self).__init__()
        self.original_model = original_model
        # 768 is the embedding dims for the original gte-base model. 
        # Adding another layer on the end to project to the output dim if needed
        if output_dim == 768:
            self.projection = torch.nn.Identity()
            for param in self.projection.parameters():
                param.requires_grad = False
        else:
            self.projection = torch.nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask=None):
        # Original model output
        outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self._average_pool(outputs.last_hidden_state, attention_mask)
        # Project to new output dim
        projected_output = self.projection(pooled_output)
        return projected_output
    
    # This function is from https://huggingface.co/thenlper/gte-base
    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    

class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset = load_dataset(path="nlphuji/flickr30k", cache_dir="./huggingface_data")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.cap_per_image = 2

    def __len__(self):
        return self.dataset.num_rows["test"] * self.cap_per_image

    def __getitem__(self, idx):
        original_idx = idx // self.cap_per_image
        image = self.dataset["test"][original_idx]["image"].convert("RGB")
        image = self.transform(image)
        caption = self.dataset["test"][original_idx]["caption"][idx % self.cap_per_image]
        return {"image": image, "caption": caption}


def freeze_pretrained_weights(model: nn.Module):
    '''
    Freezes the pretrained weights for an instance of the CustomTextEmbeddingModel class
    '''
    for param in model.original_model.parameters():
        param.requires_grad = False
    # Turn the gradients back on for the final 2 linear layers
    model.original_model.encoder.layer[10].output.requires_grad = True
    model.original_model.encoder.layer[11].output.requires_grad = True
    model.original_model.pooler.requires_grad = True


def get_text_embedding_model(model_name = "thenlper/gte-base", output_dim = 768):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = CustomTextEmbeddingModel(model, output_dim)
    freeze_pretrained_weights(model)
    model.to(device)
    return model, tokenizer


def get_image_embedding_model(model_name = "google/vit-base-patch16-224"):
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.classifier = nn.Identity()
    model.to(device)
    return model, processor


def train():
    text_model, tokenizer = get_text_embedding_model()
    image_model, processor = get_image_embedding_model()
    dataset = Flickr30kDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    loss_fn = nn.CosineSimilarity()
    optimizer = torch.optim.Adam(list(text_model.parameters()) + list(image_model.parameters()), lr=1e-4)
    training_losses = []

    # Training loop
    c = 0
    for epoch in range(1):
        for batch in dataloader:
            # Get the image and caption from the batch
            images = batch["image"].to(device)
            captions = batch["caption"]

            # Get the image embeddings
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            image_embeddings = image_model(**image_inputs).logits

            # Get the text embeddings
            text_inputs = tokenizer(captions, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device)
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            text_embeddings = text_model(input_ids, attention_mask)

            # Calculate the loss
            target = torch.ones(1).to(device)
            loss = loss_fn(text_embeddings, image_embeddings, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
            
            c += 1
            if c%100 == 0:
                print(f"Batch {c}, Loss: {loss.item()}")

        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    # Save the model
    torch.save(text_model.state_dict(), "text_model.pth")
    return training_losses


if __name__ == "__main__":
    training_losses = train()
    