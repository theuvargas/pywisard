from Wisard import Wisard, BloomWisard, DictWisard
from BloomFilter import bloom_filter_parameters
from utils import get_mnist
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import foolbox as fb

# --- Configuration ---
# Wisard Params
N_TUPLES = 48
N_NODES = 50
BITS_PER_INPUT = 6
WITH_BLEACHING = False
N_JOBS = -1  # Use all available cores

# Surrogate Model Params
SURROGATE_HIDDEN_SIZE1 = 256  # Example size
SURROGATE_HIDDEN_SIZE2 = 128  # Example size
SURROGATE_LR = 1e-3
SURROGATE_EPOCHS = 20
BATCH_SIZE = 128
TEMPERATURE = 1.0  # Temperature for softening Wisard probabilities

# Foolbox Attack Params
# Epsilon for MNIST (0-255 range, but we scale to 0-1 for NN)
# Let's use a common epsilon for [0, 1] scaled data
EPSILON = 3
ATTACK = fb.attacks.mi_fgsm.L2MomentumIterativeFastGradientMethod()  # 30.68% (eps=3)
# ATTACK = fb.attacks.L2AdamPGD() # 27.16% (eps=3)
# ATTACK = fb.attacks.L2PGD() # 20.13% (eps=3)
# ATTACK = fb.attacks.mi_fgsm.LinfMomentumIterativeFastGradientMethod()  # 68% (eps=0.15)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate_soft_targets(wisard: Wisard, X_train):
    print("\nGenerating surrogate soft targets using Wisard predict_proba...")
    wisard_proba_list = wisard.predict_proba(X_train)
    num_samples = len(wisard_proba_list)
    # Ensure classes are sorted for consistent indexing
    sorted_classes = sorted(list(wisard.classes))
    class_to_index = {c: i for i, c in enumerate(sorted_classes)}
    num_classes = len(sorted_classes)

    # Initialize target tensor (using float for scores/probabilities)
    y_wisard_scores_np = np.zeros((num_samples, num_classes), dtype=np.float32)
    for i, sample_scores_dict in enumerate(wisard_proba_list):
        for class_label, score in sample_scores_dict.items():
            if class_label in class_to_index:
                idx = class_to_index[class_label]
                y_wisard_scores_np[i, idx] = float(score)  # Ensure float

    # Convert raw scores to PyTorch tensor before Softmax
    y_wisard_scores_torch = torch.tensor(y_wisard_scores_np, dtype=torch.float32)

    # Apply Softmax along the class dimension (dim=1)
    y_surrogate_target_soft_torch = F.softmax(
        y_wisard_scores_torch / TEMPERATURE, dim=1
    )

    return y_surrogate_target_soft_torch


def train_wisard(wisard: Wisard, X_train, y_train):
    print("Training...")
    start = time.time()
    wisard.train(X_train, y_train)
    print(f"Training time: {time.time() - start}")
    return wisard


def train_surrogate(
    model, X_train_tensor, soft_targets_tensor, epochs, lr, batch_size, temp
):
    """Trains the surrogate model using soft targets and KL Divergence."""
    print("\nTraining surrogate model...")
    model.to(device)
    model.train()

    train_dataset = TensorDataset(X_train_tensor, soft_targets_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False,
    )

    # KL Divergence Loss for Distillation
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (inputs, soft_targets) in enumerate(train_loader):
            inputs, soft_targets = inputs.to(device), soft_targets.to(device)

            optimizer.zero_grad()
            outputs_logits = model(inputs)

            # Apply LogSoftmax to model outputs (required by KLDivLoss)
            # Use the same temperature as for target generation
            outputs_log_probs = F.log_softmax(outputs_logits / temp, dim=1)

            # Calculate KL Divergence (target should be probabilities)
            loss = criterion(outputs_log_probs, soft_targets)

            # Scale loss by T*T (common practice)
            loss = loss * (temp**2)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # KLDivLoss is already averaged by batchmean

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Avg KL Div Loss: {avg_epoch_loss:.6f}")

    end_time = time.time()
    print(f"Surrogate training finished in {end_time - start_time:.2f} seconds.")
    return model


class SurrogateMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.2),  # Optional dropout
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Optional dropout
            nn.Linear(hidden_size2, num_classes),
            # Output raw logits
        )

    def forward(self, x):
        return self.network(x)


def main():
    size, hashes = bloom_filter_parameters(6000, 0.05)
    wisard = BloomWisard(
        n_tuples=N_TUPLES,
        n_nodes=N_NODES,
        bits_per_input=BITS_PER_INPUT,
        bloom_size=size,
        num_hashes=hashes,
        dtype=bool,
        withBleaching=WITH_BLEACHING,
        n_jobs=N_JOBS,
    )

    X_train, y_train, X_test, y_test = get_mnist()

    train_wisard(wisard, X_train, y_train)

    # Evaluate Wisard base accuracy on test set
    print("\nEvaluating Wisard base performance...")
    y_pred_wisard = wisard.predict(X_test)
    wisard_clean_accuracy = np.mean(y_pred_wisard == y_test)
    print(f"Wisard clean accuracy on test set: {wisard_clean_accuracy:.4f}")

    # augment the training data with noise
    # augmented_X_train = np.zeros((X_train.shape[0] * 2, 784), dtype=np.float32)
    # augmented_X_train[: X_train.shape[0]] = X_train
    # augmented_X_train[X_train.shape[0] :] = np.clip(
    #     X_train + np.random.normal(0, 0.3), 0, 1
    # )
    # X_train = augmented_X_train
    # y_train = np.concatenate([y_train, y_train])

    # 3. Generate Soft Targets from Wisard for Surrogate Training
    # Use the *training* data (scaled for NN) to get targets

    soft_targets_tensor = generate_soft_targets(
        wisard, X_train
    )  # Pass original data to Wisard

    # 4. Define and Train Surrogate Model
    surrogate_model = SurrogateMLP(
        input_size=784,
        hidden_size1=SURROGATE_HIDDEN_SIZE1,
        hidden_size2=SURROGATE_HIDDEN_SIZE2,
        num_classes=10,
    )

    X_train_tensor = torch.tensor(
        X_train, dtype=torch.float32
    )  # Use scaled data for NN
    surrogate_model = train_surrogate(
        surrogate_model,
        X_train_tensor,
        soft_targets_tensor,
        SURROGATE_EPOCHS,
        SURROGATE_LR,
        BATCH_SIZE,
        TEMPERATURE,
    )

    surrogate_model.eval()
    surrogate_model.to(device)

    # with torch.no_grad():
    #     X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    #     print("\nEvaluating Surrogate model performance...")
    #     y_pred_surrogate = surrogate_model(X_test_tensor)
    #     y_pred_surrogate = torch.argmax(y_pred_surrogate, dim=1).cpu().numpy()
    #     surrogate_accuracy = np.mean(y_pred_surrogate == y_test)
    #     surrogate_wisard_accuracy = np.mean(y_pred_surrogate == y_pred_wisard)
    # print(f"Surrogate model accuracy on test set: {surrogate_accuracy:.4f}")
    # print(
    #     f"Surrogate model accuracy on Wisard predictions: {surrogate_wisard_accuracy:.4f}"
    # )

    fb_model = fb.models.pytorch.PyTorchModel(
        surrogate_model, bounds=(0, 1), device=device
    )

    correct_mask_wisard = y_pred_wisard == y_test

    X_test_correct = X_test[correct_mask_wisard]
    y_test_correct = y_test[correct_mask_wisard]

    print(
        f"Selected {len(X_test_correct)} test samples correctly classified by Wisard for attack."
    )
    # -----------------------------------------------------
    inputs_torch = torch.tensor(X_test_correct, dtype=torch.float32).to(device)
    labels_torch = torch.tensor(y_test_correct, dtype=torch.long).to(
        device
    )  # True labels

    # Choose attack and criterion
    criterion_fb = fb.criteria.Misclassification(labels_torch)

    # 6. Generate Adversarial Examples against Surrogate
    print(f"Generating adversarial examples using {ATTACK} with epsilon={EPSILON}...")
    start_attack_time = time.time()
    # Note: epsilons expects a list
    raw_advs, clipped_advs, success = ATTACK(
        fb_model, inputs_torch, criterion_fb, epsilons=[EPSILON]
    )
    end_attack_time = time.time()
    print(
        f"Foolbox attack generation took {end_attack_time - start_attack_time:.2f} seconds."
    )

    # Use the results for the single epsilon
    adv_inputs_torch = clipped_advs[0]  # Adversarial examples clipped to bounds [0, 1]
    success_mask = success[
        0
    ]  # Boolean mask: True if attack succeeded against surrogate

    surrogate_attack_success_rate = success_mask.float().mean().item()
    print(
        f"Attack success rate against *surrogate* model: {surrogate_attack_success_rate:.4f}"
    )

    # Get adversarial examples that *successfully fooled the surrogate*
    successful_adv_inputs_torch = adv_inputs_torch[success_mask]
    # Get the original labels corresponding to these successful attacks
    original_labels_for_successful = labels_torch[success_mask]

    if successful_adv_inputs_torch.shape[0] == 0:
        print(
            "No adversarial examples successfully fooled the surrogate model. Cannot test transfer."
        )
        return

    successful_adv_inputs_np = (
        successful_adv_inputs_torch.cpu().numpy().astype(X_train.dtype)
    )
    original_labels_np = original_labels_for_successful.cpu().numpy()

    print(
        f"Generated {len(successful_adv_inputs_np)} successful adversarial examples for transfer testing."
    )

    # 7. Test Transferability to Original Wisard Model
    print("\nTesting transferability to the original Wisard model...")
    start_transfer_time = time.time()
    # Predict using Wisard on the successful adversarial examples (rescaled)
    y_pred_wisard_adv = wisard.predict(successful_adv_inputs_np)
    end_transfer_time = time.time()
    print(
        f"Wisard prediction on adversarial examples took {end_transfer_time - start_transfer_time:.2f} seconds."
    )

    # Compare Wisard's predictions on adversarial examples to the *original* labels
    transfer_misclassified_mask = y_pred_wisard_adv != original_labels_np
    transfer_attack_success_rate = np.mean(transfer_misclassified_mask)
    num_transferred = np.sum(transfer_misclassified_mask)

    # --- Final Results ---
    print("\n" + "=" * 30 + " Results " + "=" * 30)
    print(f"Wisard Clean Accuracy (Test Set):         {wisard_clean_accuracy:.4f}")
    print(
        f"Surrogate Attack Success Rate (on itself): {surrogate_attack_success_rate:.4f}"
    )
    print(
        f"** Transfer Attack Success Rate to Wisard: {transfer_attack_success_rate:.4f} **"
    )
    print(
        f"   ({num_transferred} out of {len(successful_adv_inputs_np)} successful surrogate attacks transferred)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
