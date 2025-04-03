import torch
import foolbox as fb
from pywisard.attacks.models import SurrogateMLP

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wisard Parameters
N_TUPLES = 48
N_NODES = 50
BITS_PER_INPUT = 6
WITH_BLEACHING = False
N_JOBS = -1

# Surrogate Model Parameters
SURROGATE_MODEL = SurrogateMLP  # Direct reference to the model class
SURROGATE_HIDDEN_SIZE1 = 256
SURROGATE_HIDDEN_SIZE2 = 128
SURROGATE_LR = 1e-3
SURROGATE_EPOCHS = 1
BATCH_SIZE = 128
TEMPERATURE = 1.0

# Attack Parameters
EPSILON = 3
ATTACK = fb.attacks.mi_fgsm.L2MomentumIterativeFastGradientMethod()
ADVERSARIAL_TRAINING = 0
