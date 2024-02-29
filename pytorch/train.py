import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import NeuralNet
from dataset import DogsvsCatsDataset
import os
import zipfile

# HyperParams
base = "./data/dogs-vs-cats.zip"
test_dir = "./data/test1.zip"
train_dir = "./data/train.zip"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_save_path = "./pytorch/trained.pth"
lr = 1e-4
epochs = 30
# ---------


def unzipFiles(dir):
    if not os.path.exists(dir):
        with zipfile.ZipFile(dir, 'r') as z:
            z.extractall("./data")


def train(model, train_dl, num_epochs=epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_batch_loss = 0.0
        for i, (image, label) in enumerate(train_dl):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()  # backward propergation
            optimizer.step()

            running_batch_loss += loss.item()

            if i % 2000 == 1999:
                print(f"Epoch: {epoch}, Loss: {running_batch_loss / 2000:.4f}")
                running_batch_loss = 0.0


if __name__ == "__main__":
    unzipFiles(base)
    unzipFiles(train_dir)
    unzipFiles(test_dir)
    train_ds = DogsvsCatsDataset(os.path.splitext(train_dir)[0])
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = NeuralNet(3, 2).to(device)

    print("training: ")
    train(model, train_dl)

    torch.save({
        'model_state_dict': model.state_dict(),
    }, model_save_path)
