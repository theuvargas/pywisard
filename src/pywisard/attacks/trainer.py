import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import foolbox as fb
from pywisard.attacks.config import (
    SURROGATE_EPOCHS,
    SURROGATE_LR,
    TEMPERATURE,
    BATCH_SIZE,
    ATTACK,
    EPSILON,
)


class SurrogateTrainer:
    def __init__(self, wisard, surrogate_model, device):
        self.wisard = wisard
        self.surrogate_model = surrogate_model
        self.device = device
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = optim.Adam(surrogate_model.parameters(), lr=SURROGATE_LR)

    def generate_soft_targets(self, X_train):
        print("\nGenerating surrogate soft targets using Wisard predict_proba...")
        wisard_proba_list = self.wisard.predict_proba(X_train)

        # Initialize arrays for soft targets
        num_samples = len(wisard_proba_list)
        sorted_classes = sorted(list(self.wisard.classes))
        class_to_index = {c: i for i, c in enumerate(sorted_classes)}
        num_classes = len(sorted_classes)

        y_wisard_scores = np.zeros((num_samples, num_classes), dtype=np.float32)
        for i, sample_scores in enumerate(wisard_proba_list):
            for class_label, score in sample_scores.items():
                if class_label in class_to_index:
                    idx = class_to_index[class_label]
                    y_wisard_scores[i, idx] = float(score)

        scores_tensor = torch.tensor(y_wisard_scores, dtype=torch.float32)
        return F.softmax(scores_tensor / TEMPERATURE, dim=1)

    def train_epoch(self, train_loader):
        self.surrogate_model.train()
        epoch_loss = 0.0

        for inputs, soft_targets in train_loader:
            inputs = inputs.to(self.device)
            soft_targets = soft_targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.surrogate_model(inputs)
            outputs_log_probs = F.log_softmax(outputs / TEMPERATURE, dim=1)

            loss = self.criterion(outputs_log_probs, soft_targets)
            loss = loss * (TEMPERATURE**2)

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    def train(self, X_train, epochs=SURROGATE_EPOCHS):
        print("\nTraining surrogate model...")
        soft_targets = self.generate_soft_targets(X_train)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, soft_targets)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False,
        )

        start_time = time.time()
        for epoch in range(epochs):
            avg_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Avg KL Div Loss: {avg_loss:.6f}")

        print(f"Training finished in {time.time() - start_time:.2f} seconds.")

    def adversarial_training_iteration(self, X_test_correct, y_test_correct):
        """Performs one iteration of adversarial training"""
        # Setup for foolbox attack
        fb_model = fb.models.pytorch.PyTorchModel(
            self.surrogate_model, bounds=(0, 1), device=self.device
        )

        inputs_torch = torch.tensor(X_test_correct, dtype=torch.float32).to(self.device)
        labels_torch = torch.tensor(y_test_correct, dtype=torch.long).to(self.device)
        criterion_fb = fb.criteria.Misclassification(labels_torch)

        # Generate adversarial examples
        self.surrogate_model.eval()
        print(
            f"Generating adversarial examples using {ATTACK} with epsilon={EPSILON}..."
        )
        _, clipped_advs, success = ATTACK(
            fb_model, inputs_torch, criterion_fb, epsilons=[EPSILON]
        )

        # Filter successful attacks
        success_mask = success[0]
        successful_advs = clipped_advs[0][success_mask]

        if len(successful_advs) == 0:
            print("No successful adversarial examples generated. Stopping training.")
            return False

        # Add successful adversarial examples to training data
        adv_inputs_np = successful_advs.cpu().numpy()
        combined_X = np.concatenate([X_test_correct, adv_inputs_np])

        # Retrain the model with the augmented dataset
        self.train(combined_X)
        return True
