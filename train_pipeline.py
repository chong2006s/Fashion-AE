import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from autoencoder import Autoencoder
from classifier import LatentClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

autoencoder = Autoencoder().to(device)
classifier = LatentClassifier().to(device)

recon_loss_fn = nn.MSELoss()
class_loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(list(autoencoder.parameters()) + list(classifier.parameters()), lr=1e-3)

epochs = 5
for epoch in range(epochs):
    autoencoder.train()
    classifier.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        recon, latent = autoencoder(images)
        outputs = classifier(latent)

        recon_loss = recon_loss_fn(recon, images)
        class_loss = class_loss_fn(outputs, labels)
        loss = recon_loss + class_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

torch.save(autoencoder.state_dict(), "autoencoder.pth")
torch.save(classifier.state_dict(), "classifier.pth")
