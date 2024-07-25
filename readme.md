# Neural Style Transfer

This project implements Neural Style Transfer (NST) using a pre-trained VGG19 model. The code is based on a tutorial by Aladdin Pearson and adapts the method to transfer the style of one image to the content of another.

## Prerequisites

To run this project, ensure you have the following installed:

- Python 3.x
- PyTorch
- torchvision
- Pillow

## Setup

1. **Clone the repository:**

   ```
   git clone https://github.com/Ashmit110/neural-style-transfer.git
   cd neural-style-transfer
   ```
2. **Install dependencies:**

   Use `pip` to install the required Python packages:

   ```
   pip install torch torchvision pillow
   ```
3. **Download or prepare your images:**

   Place your content and style images in the specified directory. For this example, the content image is `annahathaway.png`, and the style image is `style3.webp`. Modify the paths in the code if needed.

## How to Run

1. **Modify the image paths:**

   Ensure that the paths to the content and style images are correct in the script:

   ```python
   original_img = load_image(r"C:\python learning\neural style transfer\annahathaway.png")
   style_img = load_image(r"C:\python learning\neural style transfer\style3.webp")
   ```
2. **Adjust Hyperparameters:**

   You can tweak the following hyperparameters to suit your preferences:

   - `total_steps`: Number of optimization steps (default is 6000).
   - `learning_rate`: Learning rate for the optimizer (default is 0.001).
   - `alpha`: Weight for content loss (default is 1).
   - `beta`: Weight for style loss (default is 0.01).
3. **Run the script:**

   Execute the Python script to start the style transfer process:

   ```
   python neural_style_transfer.py
   ```

   The script will print the total loss at regular intervals and save intermediate results to the `results-3` directory.

## Results

Generated images are saved every 200 steps in the `results-3` directory. You can view these images to observe the progression of style transfer.

## Acknowledgments

This project follows the tutorial by Aladdin Pearson, which can be found [here](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/tutorials/14_neural_style_transfer).
