import torch
import matplotlib.pyplot as plt
from model import NeuralNet
from dataset import DogsvsCatsDataset
from torch.utils.data import DataLoader
import os


# Hyperparams
device = "cuda" if torch.cuda.is_available() else "cpu"
model_save_path = "./pytorch/trained.pth"
test_dir = "./data/test1"
# ----------


def draw_plt(model, test_dl, num_plots=20):
    classes = ["dog", "cat"]

    images = []
    all_preds = []

    for i, (batch_images, _) in enumerate(test_dl):
        batch_images = batch_images.to(device)
        labels = model(batch_images)
        preds = torch.argmax(labels, 1)

        images.append(batch_images.cpu())
        all_preds.extend(preds.cpu().numpy())

        if i == num_plots + 1:
            break

    images = torch.cat(images, dim=0)

    all_preds = torch.tensor(all_preds)

    fig, axes = plt.subplots(num_plots // 10, 10, figsize=(20, 5))
    fig.suptitle("Predictions")
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_title(f'Prediction: {classes[all_preds[i].item()]}')
        ax.axis('off')

    plt.show()


if __name__ == '__main__':
    model = NeuralNet(3, 2)
    model.load_state_dict(torch.load(model_save_path)["model_state_dict"])

    test_ds = DogsvsCatsDataset(os.path.splitext(test_dir)[0])
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=True)

    draw_plt(model, test_dl)
