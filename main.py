import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter #to print to tensorboard

# Define the VGG model class to extract features from specific layers
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # The layers of interest based on the NST paper
        self.chosen_features = ["0", "5", "10", "19", "28"]
        
        # Use the pre-trained VGG19 model from torchvision
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []  # To store features from chosen layers

        # Pass the input through each layer of the VGG model
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)  # Save features from chosen layers

        return features

# Function to load and preprocess an image
def load_image(image_path):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image size for resizing (can be adjusted)
imsize = 356

# Define the image transformation pipeline
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # Resize the image
    transforms.ToTensor(),  # Convert image to PyTorch tensor
])

# Load the content and style images
original_img = load_image(r"K:\python learning\neural style transfer\annahathaway.png")
style_img = load_image(r"K:\python learning\neural style transfer\style3.webp")

# Initialize the generated image as a copy of the content image
generated = original_img.clone().requires_grad_(True)

# Initialize the VGG model
model = VGG().to(device).eval()

# Hyperparameters
total_steps = 6000  # Total number of optimization steps
learning_rate = 0.001  # Learning rate for the optimizer
alpha = 1  # Weight for content loss
beta = 0.01  # Weight for style loss

# Set up the optimizer to update the generated image
optimizer = optim.Adam([generated], lr=learning_rate)
writer=SummaryWriter(f'runs/test/tryingout_tensorboard')

# Optimization loop
for step in range(total_steps):
    # Extract features for content, style, and generated images
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    # Initialize loss values
    style_loss = 0
    content_loss = 0

    # Calculate the style and content loss
    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_img_features, style_features
    ):
        # Calculate content loss
        content_loss += torch.mean((gen_feature - orig_feature) ** 2)

        # Calculate Gram matrix for style features
        batch_size, channel, height, width = gen_feature.shape
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )
        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        # Calculate style loss
        style_loss += torch.mean((G - A) ** 2)

    # Calculate total loss as a weighted sum of content and style loss
    total_loss = alpha * content_loss + beta * style_loss

    # Backpropagation and optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    writer.add_scalar('training loss', total_loss.item(), global_step=step)
    # writer.add_scalar('traing accuracy,')
    
    # Print and save the generated image at regular intervals
    if step % 200 == 0:
        writer.add_image('Generated Image', generated.squeeze(0), global_step=step)
        print(f"Step [{step}/{total_steps}], Total Loss: {total_loss.item():.4f}")
        # save_image(generated, f"C:\\python learning\\neural style transfer\\results-3\\{step}.png")
