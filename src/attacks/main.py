from Wisard import BloomWisard
from BloomFilter import bloom_filter_parameters
from utils import get_mnist
from trainer import SurrogateTrainer
from config import (
    N_TUPLES,
    N_NODES,
    BITS_PER_INPUT,
    WITH_BLEACHING,
    N_JOBS,
    DEVICE,
    SURROGATE_MODEL,
    SURROGATE_HIDDEN_SIZE1,
    SURROGATE_HIDDEN_SIZE2,
    ADVERSARIAL_TRAINING,
    ATTACK,
    EPSILON,
)
import foolbox as fb
import torch
import numpy as np
import time


def evaluate_wisard_base(wisard, X_test, y_test):
    print("\nEvaluating Wisard base performance...")
    y_pred_wisard = wisard.predict(X_test)
    wisard_clean_accuracy = np.mean(y_pred_wisard == y_test)
    print(f"Wisard clean accuracy on test set: {wisard_clean_accuracy:.4f}")
    return y_pred_wisard, wisard_clean_accuracy


def main():
    # Initialize Wisard
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

    # Load and prepare data
    X_train, y_train, X_test, y_test = get_mnist()

    # Train Wisard
    print("Training Wisard...")
    start = time.time()
    wisard.train(X_train, y_train)
    print(f"Training time: {time.time() - start}")

    # Evaluate Wisard
    y_pred_wisard, wisard_clean_accuracy = evaluate_wisard_base(wisard, X_test, y_test)

    # Initialize surrogate model
    surrogate_model = SURROGATE_MODEL(
        input_size=784,
        hidden_size1=SURROGATE_HIDDEN_SIZE1,
        hidden_size2=SURROGATE_HIDDEN_SIZE2,
        num_classes=10,
    ).to(DEVICE)

    # Initialize trainer
    trainer = SurrogateTrainer(wisard, surrogate_model, DEVICE)

    # Initial training
    trainer.train(X_train)

    # Get correctly classified samples for attack
    correct_mask_wisard = y_pred_wisard == y_test
    X_test_correct = X_test[correct_mask_wisard]
    y_test_correct = y_test[correct_mask_wisard]

    print(
        f"Selected {len(X_test_correct)} test samples correctly classified by Wisard for attack."
    )

    # Perform adversarial training iterations
    for iteration in range(ADVERSARIAL_TRAINING):
        print(
            f"\nStarting adversarial training iteration {iteration + 1}/{ADVERSARIAL_TRAINING}"
        )
        success = trainer.adversarial_training_iteration(X_test_correct, y_test_correct)

        if not success:
            print("Aborting adversarial training due to no successful attacks.")
            break

    # Final evaluation
    surrogate_model.eval()
    fb_model = fb.models.pytorch.PyTorchModel(
        surrogate_model, bounds=(0, 1), device=DEVICE
    )

    # Final attack
    inputs_torch = torch.tensor(X_test_correct, dtype=torch.float32).to(DEVICE)
    labels_torch = torch.tensor(y_test_correct, dtype=torch.long).to(DEVICE)
    criterion_fb = fb.criteria.Misclassification(labels_torch)

    print("\nGenerating final adversarial examples...")
    _, clipped_advs, success = ATTACK(
        fb_model, inputs_torch, criterion_fb, epsilons=[EPSILON]
    )

    success_mask = success[0]
    successful_advs = clipped_advs[0][success_mask]
    original_labels = y_test_correct[success_mask]

    if len(successful_advs) == 0:
        print("No successful adversarial examples in final attack.")
        return

    # Test transferability
    successful_advs_np = successful_advs.cpu().numpy()
    y_pred_wisard_adv = wisard.predict(successful_advs_np)
    transfer_success_rate = np.mean(y_pred_wisard_adv != original_labels)

    # Print final results
    print("\n" + "=" * 30 + " Final Results " + "=" * 30)
    print(f"Wisard Clean Accuracy: {wisard_clean_accuracy:.4f}")
    print(f"Final Transfer Attack Success Rate: {transfer_success_rate:.4f}")
    print(
        f"Number of successful transfers: {np.sum(y_pred_wisard_adv != original_labels)} "
        f"out of {len(successful_advs_np)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
