import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
epochs = 20
image_size = 28  # MNIST image size
latent_dim = 20  # Size of latent space
message_length = 32  # Length of binary message to hide

# Load MNIST dataset (simple dataset to start with)
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class ImageSteganographyVAE(nn.Module):
    def __init__(self, image_size, latent_dim, message_length):
        super(ImageSteganographyVAE, self).__init__()
        
        # Image dimensions
        self.image_size = image_size
        self.image_channels = 1  # Grayscale for MNIST
        self.hidden_dim = 400  # Hidden dimension
        self.latent_dim = latent_dim
        self.message_length = message_length
        
        # Message encoder (transforms binary message for embedding)
        self.msg_preprocessor = nn.Sequential(
            nn.Linear(self.message_length, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.image_size * self.image_size)  # Same size as flattened image
        )
        
        # Encoder (image with embedded message to latent space)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_size * self.image_size * self.image_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        
        # Mean and variance for VAE
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, self.latent_dim)
        
        # Message decoder (extracts message from stego-image)
        self.msg_decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_size * self.image_size * self.image_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.message_length),
            nn.Sigmoid()  # Output probabilities for binary message
        )
        
        # Image decoder (latent space to image)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.image_size * self.image_size * self.image_channels),
            nn.Sigmoid()  # Image pixel values between 0 and 1
        )
    
    def embed_message(self, x, msg):
        """Embed the message directly into the image"""
        batch_size = x.size(0)
        
        # Preprocess message to match image size
        msg_processed = self.msg_preprocessor(msg)
        msg_processed = msg_processed.view(batch_size, 1, self.image_size, self.image_size)
        
        # Scale the message contribution to be small (strength parameter can be adjusted)
        strength = 0.1
        msg_processed = msg_processed * strength
        
        # Add the message pattern to the original image
        # Using addition with small strength to minimally impact visual appearance
        stego_image = x + msg_processed
        
        # Ensure pixel values remain in valid range [0, 1]
        stego_image = torch.clamp(stego_image, 0, 1)
        
        return stego_image
    
    def extract_message(self, stego_image):
        """Extract the message from the stego image"""
        msg_pred = self.msg_decoder(stego_image)
        return msg_pred
    
    def encode(self, x):
        """Encode the image into latent space parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode the latent representation back to image"""
        return self.decoder(z).view(-1, self.image_channels, self.image_size, self.image_size)
    
    def forward(self, x, msg):
        """Forward pass through the network"""
        # Embed message in the original image
        stego_image = self.embed_message(x, msg)
        
        # Extract message from stego image (for training the extraction)
        msg_pred = self.extract_message(stego_image)
        
        # Encode stego image to latent space
        mu, log_var = self.encode(stego_image)
        z = self.reparameterize(mu, log_var)
        
        # Decode image from latent space
        x_recon = self.decode(z)
        
        return stego_image, x_recon, msg_pred, mu, log_var

def loss_function(stego_image, original_image, x_recon, msg_pred, msg, mu, log_var, 
                  stego_weight=1.0, msg_weight=1.0, kl_weight=0.1):
    """
    Calculate the loss:
    - Steganography loss (difference between original and stego image)
    - Reconstruction loss for the image
    - Binary cross-entropy for the message
    - KL divergence for the VAE regularization
    """
    # Steganography loss (MSE between original and stego image)
    stego_loss = F.mse_loss(stego_image, original_image, reduction='sum') * stego_weight
    
    # Image reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, original_image, reduction='sum')
    
    # Message reconstruction loss (Binary Cross Entropy)
    msg_loss = F.binary_cross_entropy(msg_pred, msg, reduction='sum') * msg_weight
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * kl_weight
    
    # Total loss
    total_loss = stego_loss + recon_loss + msg_loss + kl_loss
    
    return total_loss, stego_loss, recon_loss, msg_loss, kl_loss

# Function to generate random binary messages
def generate_message(batch_size, message_length):
    """Generate random binary messages"""
    msg = torch.randint(0, 2, (batch_size, message_length), dtype=torch.float32)
    return msg

# Training function
def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    stego_loss_sum = 0
    recon_loss_sum = 0
    msg_loss_sum = 0
    kl_loss_sum = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Generate random messages for training
        msg = generate_message(data.size(0), model.message_length).to(device)
        
        # Forward pass
        stego_image, x_recon, msg_pred, mu, log_var = model(data, msg)
        
        # Calculate loss
        loss, stego_loss, recon_loss, msg_loss, kl_loss = loss_function(
            stego_image, data, x_recon, msg_pred, msg, mu, log_var)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        train_loss += loss.item()
        stego_loss_sum += stego_loss.item()
        recon_loss_sum += recon_loss.item()
        msg_loss_sum += msg_loss.item()
        kl_loss_sum += kl_loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    # Print average losses
    avg_loss = train_loss / len(train_loader.dataset)
    avg_stego_loss = stego_loss_sum / len(train_loader.dataset)
    avg_recon_loss = recon_loss_sum / len(train_loader.dataset)
    avg_msg_loss = msg_loss_sum / len(train_loader.dataset)
    avg_kl_loss = kl_loss_sum / len(train_loader.dataset)
    
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.6f}, '
          f'Stego: {avg_stego_loss:.6f}, Recon: {avg_recon_loss:.6f}, '
          f'Msg: {avg_msg_loss:.6f}, KL: {avg_kl_loss:.6f}')
    
    return avg_loss, avg_stego_loss, avg_recon_loss, avg_msg_loss, avg_kl_loss

# Test function
def test(model, test_loader):
    model.eval()
    test_loss = 0
    stego_loss_sum = 0
    recon_loss_sum = 0
    msg_loss_sum = 0
    kl_loss_sum = 0
    msg_accuracy_sum = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # Generate random messages for testing
            msg = generate_message(data.size(0), model.message_length).to(device)
            
            # Forward pass
            stego_image, x_recon, msg_pred, mu, log_var = model(data, msg)
            
            # Calculate loss
            loss, stego_loss, recon_loss, msg_loss, kl_loss = loss_function(
                stego_image, data, x_recon, msg_pred, msg, mu, log_var)
            
            # Update metrics
            test_loss += loss.item()
            stego_loss_sum += stego_loss.item()
            recon_loss_sum += recon_loss.item()
            msg_loss_sum += msg_loss.item()
            kl_loss_sum += kl_loss.item()
            
            # Calculate message accuracy (threshold at 0.5)
            msg_pred_binary = (msg_pred > 0.5).float()
            accuracy = (msg_pred_binary == msg).float().mean(dim=1)
            msg_accuracy_sum += accuracy.sum().item()
    
    # Calculate average metrics
    avg_loss = test_loss / len(test_loader.dataset)
    avg_stego_loss = stego_loss_sum / len(test_loader.dataset)
    avg_recon_loss = recon_loss_sum / len(test_loader.dataset) 
    avg_msg_loss = msg_loss_sum / len(test_loader.dataset)
    avg_kl_loss = kl_loss_sum / len(test_loader.dataset)
    avg_msg_accuracy = msg_accuracy_sum / len(test_loader.dataset)
    
    print(f'====> Test set loss: {avg_loss:.6f}, '
          f'Stego: {avg_stego_loss:.6f}, Recon: {avg_recon_loss:.6f}, '
          f'Msg: {avg_msg_loss:.6f}, KL: {avg_kl_loss:.6f}, '
          f'Msg Accuracy: {avg_msg_accuracy:.4f}')
    
    return avg_loss, avg_msg_accuracy

# Visualization function
def visualize_results(model, test_loader):
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        data, _ = next(iter(test_loader))
        data = data.to(device)
        
        # Generate random messages
        msg = generate_message(data.size(0), model.message_length).to(device)
        
        # Forward pass
        stego_image, x_recon, msg_pred, _, _ = model(data, msg)
        
        # Convert message predictions to binary
        msg_pred_binary = (msg_pred > 0.5).float()
        
        # Move tensors to CPU for visualization
        data = data.cpu()
        stego_image = stego_image.cpu()
        x_recon = x_recon.cpu()
        msg = msg.cpu()
        msg_pred_binary = msg_pred_binary.cpu()
        
        # Create figure
        fig, axes = plt.subplots(5, 8, figsize=(16, 10))
        
        # Plot original images
        for i in range(8):
            ax = axes[0, i]
            ax.imshow(data[i].numpy().reshape(28, 28), cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title('Original')
        
        # Plot stego images
        for i in range(8):
            ax = axes[1, i]
            ax.imshow(stego_image[i].numpy().reshape(28, 28), cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title('Stego Image')
        
        # Plot reconstructed images
        for i in range(8):
            ax = axes[2, i]
            ax.imshow(x_recon[i].numpy().reshape(28, 28), cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title('Reconstructed')
        
        # Plot original messages
        for i in range(8):
            ax = axes[3, i]
            ax.imshow(msg[i].numpy().reshape(1, -1), cmap='binary', aspect='auto')
            ax.axis('off')
            if i == 0:
                ax.set_title('Original Message')
        
        # Plot reconstructed messages
        for i in range(8):
            ax = axes[4, i]
            ax.imshow(msg_pred_binary[i].numpy().reshape(1, -1), cmap='binary', aspect='auto')
            ax.axis('off')
            if i == 0:
                ax.set_title('Recovered Message')
        
        # Plot the difference between original and stego images
        difference = torch.abs(stego_image - data)
        
        # Create a new figure for the difference
        plt.figure(figsize=(16, 3))
        for i in range(8):
            plt.subplot(1, 8, i+1)
            plt.imshow(difference[i].numpy().reshape(28, 28), cmap='hot')
            plt.axis('off')
            if i == 0:
                plt.title('Difference (Message Impact)')
        
        plt.tight_layout()
        plt.savefig('steganography_difference.png')
        
        # Save the main results
        fig.tight_layout()
        fig.savefig('steganography_results.png')
        plt.close('all')
        
        print("Results visualization saved to 'steganography_results.png'")
        print("Difference visualization saved to 'steganography_difference.png'")

# Main training loop
def main():
    # Initialize model
    model = ImageSteganographyVAE(image_size=image_size, latent_dim=latent_dim, message_length=message_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Lists to store metrics
    train_losses = []
    test_losses = []
    msg_accuracies = []
    
    # Train the model
    for epoch in range(1, epochs + 1):
        train_loss, _, _, _, _ = train(model, train_loader, optimizer, epoch)
        test_loss, msg_accuracy = test(model, test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        msg_accuracies.append(msg_accuracy)
        
        # Visualize results every 5 epochs
        if epoch % 5 == 0:
            visualize_results(model, test_loader)
    
    # Save the trained model
    torch.save(model.state_dict(), 'image_steganography_vae.pth')
    print("Model saved to 'image_steganography_vae.pth'")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(msg_accuracies, label='Message Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    print("Training curves saved to 'training_curves.png'")

if __name__ == "__main__":
    main()