from Wisard import Wisard, BloomWisard, DictWisard
from BloomFilter import bloom_filter_parameters
from utils import get_mnist
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import foolbox as fb
from foolbox.criteria import Misclassification  # Import criterion directly

# Wisard Params
N_TUPLES = 48
N_NODES = 50
BITS_PER_INPUT = 4
WITH_BLEACHING = False
N_JOBS = -1  # Use all available cores

# Surrogate Model Params
SURROGATE_HIDDEN_SIZE1 = 256  # Example size
SURROGATE_HIDDEN_SIZE2 = 128  # Example size
SURROGATE_LR = 1e-3
SURROGATE_EPOCHS = 20
BATCH_SIZE = 128

# Foolbox Attack Params
EPSILON = 3  # Example for LinfMomentumIterativeFastGradientMethod
ATTACK = fb.attacks.mi_fgsm.L2MomentumIterativeFastGradientMethod()
# ATTACK = fb.attacks.L2AdamPGD() # L2 attack, might need larger EPSILON
# ATTACK = fb.attacks.L2PGD() # L2 attack, might need larger EPSILON
# ATTACK = fb.attacks.mi_fgsm.LinfMomentumIterativeFastGradientMethod()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_wisard(wisard: Wisard, X_train, y_train):
    """Trains the Wisard model."""
    print("\nTraining Wisard model...")
    start = time.time()
    wisard.train(X_train, y_train)
    end = time.time()
    print(f"Wisard training finished in {end - start:.2f} seconds.")
    return wisard


def train_surrogate_hard(
    model, X_train_tensor, hard_targets_tensor, epochs, lr, batch_size
):
    """Trains the surrogate model using hard labels and CrossEntropyLoss."""
    print("\nTraining surrogate model (using hard labels)...")
    model.to(device)
    model.train()

    # Ensure targets are of type LongTensor for CrossEntropyLoss
    if hard_targets_tensor.dtype != torch.long:
        hard_targets_tensor = hard_targets_tensor.long()

    train_dataset = TensorDataset(X_train_tensor, hard_targets_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Use CrossEntropyLoss for hard labels
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0
        for i, (inputs, hard_targets) in enumerate(train_loader):
            inputs, hard_targets = inputs.to(device), hard_targets.to(device)

            optimizer.zero_grad()
            # Get raw logits from the model
            outputs_logits = model(inputs)

            # Calculate CrossEntropyLoss (expects raw logits and long labels)
            loss = criterion(outputs_logits, hard_targets)

            loss.backward()
            optimizer.step()

            epoch_loss += (
                loss.item()
            )  # loss is already averaged over the batch by default

            # Calculate accuracy against Wisard's predictions for monitoring
            _, predicted = torch.max(outputs_logits.data, 1)
            total_preds += hard_targets.size(0)
            correct_preds += (predicted == hard_targets).sum().item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds
        # Print CrossEntropy loss and accuracy vs Wisard labels
        print(
            f"Epoch {epoch+1}/{epochs} - Avg CE Loss: {avg_epoch_loss:.6f} - Acc vs Wisard: {epoch_acc:.4f}"
        )

    end_time = time.time()
    print(f"Surrogate training finished in {end_time - start_time:.2f} seconds.")
    return model


class SurrogateMLP(nn.Module):
    """Simple MLP for surrogate modeling."""

    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.2),  # Keep dropout? Optional.
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size2, num_classes),
            # Output raw logits (for CrossEntropyLoss)
        )

    def forward(self, x):
        # Ensure input is flattened if needed (MNIST from get_mnist usually is)
        # x = x.view(x.size(0), -1) # Uncomment if input isn't already flat
        return self.network(x)


def main():
    # 1. Load Data
    print("Loading MNIST data...")
    X_train, y_train, X_test, y_test = get_mnist()
    print(f"Original data range: Min={X_train.min()}, Max={X_train.max()}")
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # 2. Initialize and Train Wisard
    try:
        # Estimate items based on expected unique patterns per node
        # This is a very rough estimate, might need tuning
        bloom_items_estimate = 10000  # Or len(X_train_wisard) * N_NODES / some_factor
        fp_rate = 0.01
        size, hashes = bloom_filter_parameters(bloom_items_estimate, fp_rate)
        print(f"Calculated Bloom parameters: size={size}, hashes={hashes}")
    except NameError:
        print("Using hardcoded Bloom parameters as function was missing.")
        size, hashes = 600000, 7

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

    wisard = train_wisard(wisard, X_train, y_train)

    # Evaluate Wisard base accuracy on test set
    print("\nEvaluating Wisard base performance...")
    y_pred_wisard_clean = wisard.predict(X_test)
    # Ensure Wisard predictions are compatible type with y_test_wisard if necessary
    try:
        y_pred_wisard_clean = y_pred_wisard_clean.astype(y_test.dtype)
    except (
        AttributeError
    ):  # Handle case where predict might not return numpy array directly
        y_pred_wisard_clean = np.array(y_pred_wisard_clean, dtype=y_test.dtype)

    wisard_clean_accuracy = np.mean(y_pred_wisard_clean == y_test)
    print(f"Wisard clean accuracy on test set: {wisard_clean_accuracy:.4f}")

    # 3. Generate HARD Targets from Wisard for Surrogate Training
    print("\nGenerating surrogate HARD targets using Wisard predict...")
    start_time = time.time()
    y_wisard_hard_pred_train = wisard.predict(X_train)
    # Ensure Wisard predictions are integers for LongTensor
    y_wisard_hard_target_tensor = torch.tensor(
        y_wisard_hard_pred_train, dtype=torch.long
    )
    print(f"Hard target generation took {time.time() - start_time:.2f} seconds.")
    print(f"Generated {y_wisard_hard_target_tensor.shape} hard target tensor.")

    # 4. Define and Train Surrogate Model using Hard Targets
    surrogate_model = SurrogateMLP(
        input_size=input_size,  # Should be 784 for flattened MNIST
        hidden_size1=SURROGATE_HIDDEN_SIZE1,
        hidden_size2=SURROGATE_HIDDEN_SIZE2,
        num_classes=num_classes,  # Should be 10 for MNIST
    )

    # Use SCALED data for surrogate training input
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    surrogate_model = train_surrogate_hard(  # Call the modified training function
        surrogate_model,
        X_train_tensor,
        y_wisard_hard_target_tensor,
        SURROGATE_EPOCHS,
        SURROGATE_LR,
        BATCH_SIZE,
    )

    # --- Foolbox Attack Section ---
    surrogate_model.eval()
    surrogate_model.to(device)  # Ensure model is on device after training

    # Wrap model for Foolbox - IMPORTANT: use bounds=[0, 1] for scaled data
    fb_model = fb.models.pytorch.PyTorchModel(
        surrogate_model, bounds=(0, 1), device=device
    )

    # Select test samples correctly classified by *Wisard*
    # Use original test labels (y_test_wisard)
    correct_mask_wisard = y_pred_wisard_clean == y_test
    # Use SCALED test data for generating attacks
    X_test_correct_nn = X_test[correct_mask_wisard]
    # Use original TRUE labels for the criterion
    y_test_correct_true = y_test[correct_mask_wisard]

    if len(X_test_correct_nn) == 0:
        print(
            "Warning: Wisard classified 0 test samples correctly. Cannot evaluate attack transfer."
        )
        return  # Exit if no samples to attack

    print(
        f"\nSelected {len(X_test_correct_nn)} test samples correctly classified by Wisard for attack."
    )

    # Convert selected samples (scaled) and TRUE labels to tensors for Foolbox
    inputs_torch = torch.tensor(X_test_correct_nn, dtype=torch.float32).to(device)
    labels_torch = torch.tensor(y_test_correct_true, dtype=torch.long).to(device)

    # Define the criterion: we want the surrogate to misclassify the TRUE label
    criterion_fb = Misclassification(labels_torch)

    # 6. Generate Adversarial Examples against Surrogate
    print(f"Generating adversarial examples using {ATTACK} with epsilon={EPSILON}...")
    start_attack_time = time.time()
    raw_advs, clipped_advs, success = ATTACK(
        fb_model, inputs_torch, criterion_fb, epsilons=[EPSILON]
    )
    end_attack_time = time.time()
    print(
        f"Foolbox attack generation took {end_attack_time - start_attack_time:.2f} seconds."
    )

    # Use results for the single epsilon
    adv_inputs_torch = clipped_advs[0]  # Adversarial examples in [0, 1] range
    success_mask = success[0]  # Success against surrogate w.r.t. TRUE labels

    surrogate_attack_success_rate = success_mask.float().mean().item()
    print(
        f"Attack success rate against *surrogate* model (vs true labels): {surrogate_attack_success_rate:.4f}"
    )

    # Filter for examples that successfully fooled the surrogate
    successful_adv_inputs_torch = adv_inputs_torch[success_mask]
    # Get the original TRUE labels corresponding to these successful attacks
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
    # Predict using Wisard on the RESCALED successful adversarial examples
    y_pred_wisard_adv = wisard.predict(successful_adv_inputs_np)
    # Ensure prediction type matches for comparison
    try:
        y_pred_wisard_adv = y_pred_wisard_adv.astype(original_labels_np.dtype)
    except AttributeError:
        y_pred_wisard_adv = np.array(y_pred_wisard_adv, dtype=original_labels_np.dtype)

    end_transfer_time = time.time()
    print(
        f"Wisard prediction on adversarial examples took {end_transfer_time - start_transfer_time:.2f} seconds."
    )

    # Compare Wisard's predictions on adversarial examples to their ORIGINAL TRUE labels
    transfer_misclassified_mask = y_pred_wisard_adv != original_labels_np
    transfer_attack_success_rate = np.mean(transfer_misclassified_mask)
    num_transferred = np.sum(transfer_misclassified_mask)

    # --- Final Results ---
    print("\n" + "=" * 30 + " Results " + "=" * 30)
    print(f"Wisard Clean Accuracy (Test Set):         {wisard_clean_accuracy:.4f}")
    print(
        f"Surrogate Attack Success Rate (on itself): {surrogate_attack_success_rate:.4f}"
    )
    print(f"Transfer Attack Success Rate to Wisard: {transfer_attack_success_rate:.4f}")
    print(
        f"({num_transferred}/{len(successful_adv_inputs_np)} successful surrogate attacks transferred)"
    )
    print("=" * 70)


if __name__ == "__main__":
    # Optional: Set seed for reproducibility if desired
    # torch.manual_seed(42)
    # np.random.seed(42)
    main()
