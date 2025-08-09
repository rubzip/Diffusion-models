# Diffusion Models

This project implements DDM paper for image generation using the CIFAR10 dataset. **This is an educational project created to understand and learn about diffusion models**, which are a class of generative models that learn to generate data through a gradual diffusion process.

Example of generated image

![https://github.com/rubzip/Diffusion-models/blob/main/imgs/generated_image.png](https://github.com/rubzip/Diffusion-models/blob/main/imgs/generated_image.png)
## Architecture

- **U-Net Architecture**: Implementation of a U-Net network for the denoising process
- **Sinusoidal Embedding**: Temporal sinusoidal encoding for the diffusion process

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Diffusion-models
```

2. Install dependencies:
```bash
pip install torch torchvision datasets matplotlib numpy
```

## Usage

### Training

To train the diffusion model:

```bash
python train.py
```

### Image Generation

To generate images with the trained model:

```bash
python generate.py imgs/image.png
```

### Configuration

You can modify parameters in `config.py`:

```python
NUM_TIMESTEPS = 200      # Diffusion steps
EPOCHS = 40             # Training epochs
BATCH_SIZE = 8          # Batch size
BETA_MIN = 1e-4         # Minimum beta for schedule
BETA_MAX = 0.02         # Maximum beta for schedule
EMBEDDING_DIM = 256     # Temporal embedding dimension
INPUT_SHAPE = (3, 32, 32)  # Input shape (CIFAR10)
```
