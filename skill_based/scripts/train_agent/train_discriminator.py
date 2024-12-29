import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from agents.models import Discriminator
import os

class DiscriminatorTrainer:
    def __init__(self, bc_traj, fake_traj, ckpt=None, regularized=False):
        self.bc_traj = bc_traj
        self.fake_traj = fake_traj
        self.regularized = regularized 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_data()
        self._prepare_data()
        self._initialize_model()
        if ckpt is not None:
            if type(ckpt) == str:
                self.discriminator.load_state_dict(torch.load(ckpt))

    def _load_data(self):
        """Loads the real trajectory embeddings from the file."""
        data = np.load(self.bc_traj, allow_pickle=True).item()
        fake_data = np.load(self.fake_traj, allow_pickle=True).item()
        self.real_embeddings = data["embeddings"]
        self.real_labels = data["labels"]
        self.fake_embeddings = fake_data["embeddings"]
        self.fake_labels = fake_data["labels"]
        # label smoothing
        # if self.regularized:
        #     #add large noise to labels
        #     noise_level = 0.1
        #     noise = np.random.normal(0, noise_level, self.real_embeddings.shape)
        #     self.real_embeddings += noise

        #     self.real_labels = np.full_like(self.real_labels, 0.9)
        #     self.fake_labels = np.full_like(self.fake_labels, 0.1)
        # print(self.real_embeddings)
        # print(self.fake_embeddings)
        # print(self.fake_labels)

    def _prepare_data(self):
        """Combines real and fake data, and prepares PyTorch tensors and DataLoader."""
        all_embeddings = np.vstack([self.real_embeddings, self.fake_embeddings])
        all_labels = np.hstack([self.real_labels, self.fake_labels])

        embeddings_tensor = torch.FloatTensor(all_embeddings)
        labels_tensor = torch.FloatTensor(all_labels)

        dataset = TensorDataset(embeddings_tensor, labels_tensor)
        self.dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    def _initialize_model(self):
        """Initializes the discriminator model, loss function, and optimizer."""
        self.input_dim = self.real_embeddings.shape[1]
        self.discriminator = Discriminator(input_dim=self.input_dim, regularized=self.regularized).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)

    def train(self, save_path, num_epochs=10):
        """Trains the discriminator model."""
        self.discriminator.train()
        iter_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_embeddings, batch_labels in self.dataloader:
                batch_embeddings, batch_labels = batch_embeddings.to(self.device), batch_labels.to(self.device)

                # Forward pass
                outputs = self.discriminator(batch_embeddings).squeeze()
                loss = self.criterion(outputs, batch_labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()
            iter_loss += epoch_loss

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(self.dataloader):.4f}")

        self.save_model(save_path)
        return iter_loss

    def save_model(self, path):
        """Saves the trained discriminator model to a file."""
        file_path = os.path.join(path,"model.pt")
        torch.save(self.discriminator.state_dict(), file_path)

    def evaluate(self):
        """Evaluates the trained model on real and fake samples."""
        self.discriminator.eval()
        with torch.no_grad():
            real_test = torch.FloatTensor(self.real_embeddings[:5]).to(self.device)
            real_preds = self.discriminator(real_test).squeeze()
            print("Real samples predictions:", real_preds.cpu().numpy())

            fake_test = torch.FloatTensor(self.fake_embeddings[:5]).to(self.device)
            fake_preds = self.discriminator(fake_test).squeeze()
            print("Fake samples predictions:", fake_preds.cpu().numpy())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a Discriminator model.")
    parser.add_argument("--bc_traj", type=str, required=True, help="Path to the real trajectory embeddings file.")
    parser.add_argument("--fake_traj", type=str, required=True, help="Path to the fake trajectory embeddings file.")
    parser.add_argument("--save_ckpt", type=str, default="ckpt/discriminator", help="Path to save the trained model.")
    parser.add_arguemnt("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint to resume training.")
    args = parser.parse_args()

    num_epochs = args.num_epochs
    if args.checkpoint is not None:
        trainer = DiscriminatorTrainer(args.bc_traj, args.fake_traj, ckpt=args.checkpoint)
    else:
        trainer = DiscriminatorTrainer(args.bc_traj, args.fake_traj)
    if not os.path.exists(args.save_ckpt):
        os.makedirs(args.save_ckpt)
    trainer.train(args.save_ckpt ,num_epochs=10)
    trainer.evaluate()
