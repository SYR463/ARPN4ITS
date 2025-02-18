import os
import numpy as np
import json
import torch


class Trainer:
    """Main class for model training"""

    def __init__(
            self,
            model,
            epochs,
            train_dataloader,
            train_steps,
            val_dataloader,
            val_steps,
            checkpoint_frequency,
            criterion,
            optimizer,
            lr_scheduler,
            device,
            model_dir,
            model_name,
    ):
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name

        self.loss = {"train": [], "val": []}
        self.model.to(self.device)

    def train_phase(self, phase=1):
        """Train the model for a specific phase (1 for Skip-Gram, 2 for CBOW)"""

        # Initialize early stopping variables
        best_val_loss = float('inf')  # Best validation loss seen so far
        epochs_without_improvement = 0  # Counter for epochs without improvement
        patience = 3  # Set the patience value (number of epochs without improvement before stopping)

        for epoch in range(self.epochs):
            # Training phase
            self._train_epoch()

            # Validation phase
            self._validate_epoch()

            # Get current validation loss
            val_loss = self.loss['val'][-1]

            # Print current epoch, training and validation losses
            print(f"Epoch: {epoch + 1}/{self.epochs}, Train Loss={self.loss['train'][-1]:.5f}, Val Loss={val_loss:.5f}")

            # Update learning rate scheduler
            self.lr_scheduler.step()

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save the model if validation loss improves
                self.save_model()
            else:
                epochs_without_improvement += 1
                print(f"Validation loss did not improve for {epochs_without_improvement} epoch(s).")

            # Check if early stopping should be triggered
            if epochs_without_improvement >= patience:
                print(f"Early stopping: Validation loss did not improve for {patience} epochs.")
                break

            # Save checkpoint at specified frequency
            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

    def _train_epoch(self):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

        return epoch_loss

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self.val_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

        return epoch_loss

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)
