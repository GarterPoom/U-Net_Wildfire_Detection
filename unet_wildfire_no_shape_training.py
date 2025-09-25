# Core Python utilities
import os  # Importing os for file and directory operations
import glob  # Importing glob for file pattern matching

# PyTorch (Deep Learning Framework)
import torch  # Importing PyTorch for tensor operations and deep learning
import torch.nn as nn  # Importing neural network modules from PyTorch
import torch.optim as optim  # Importing optimization algorithms from PyTorch

# System utilities
import platform  # Importing platform for system information
import psutil  # Importing psutil for system resource monitoring
import datetime  # Importing datetime for timestamp generation

# PyTorch Dataset utilities
from torch.utils.data import Dataset, DataLoader  # Importing Dataset and DataLoader for data handling

# Raster (geospatial imagery) processing
import rasterio  # Importing rasterio for reading and processing geospatial raster data
from rasterio.windows import Window  # Importing Window for specifying raster subsets

# Numerical & image processing
import numpy as np  # Importing NumPy for numerical operations
from skimage.transform import resize  # Importing resize for image resizing

# Progress tracking
from tqdm import tqdm  # Importing tqdm for progress bar visualization

# Data handling & visualization
import pandas as pd  # Importing pandas for data manipulation and analysis
import seaborn as sns  # Importing seaborn for statistical data visualization
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Evaluation metrics
from sklearn.model_selection import train_test_split  # Importing train_test_split for data splitting


# ------------------ U-Net Modules ------------------
class DoubleConv(nn.Module):
    # DoubleConv: Applies two consecutive convolution layers with batch normalization and ReLU activation
    def __init__(self, in_channels, out_channels):
        # Initialize DoubleConv module with input and output channels
        super().__init__()  # Call parent class (nn.Module) constructor
        self.double_conv = nn.Sequential(  # Define sequential container for double convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # First 3x3 convolution layer
            nn.BatchNorm2d(out_channels),  # Batch normalization for output channels
            nn.ReLU(inplace=True),  # ReLU activation (in-place to save memory)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Second 3x3 convolution layer
            nn.BatchNorm2d(out_channels),  # Batch normalization for output channels
            nn.ReLU(inplace=True)  # ReLU activation (in-place)
        )

    def forward(self, x):
        # Forward pass: Apply double convolution to input tensor
        return self.double_conv(x)  # Return output of double convolution

class Down(nn.Module):
    # Down: Downsampling block with max pooling followed by DoubleConv
    def __init__(self, in_channels, out_channels):
        # Initialize Down module with input and output channels
        super().__init__()  # Call parent class constructor
        self.maxpool_conv = nn.Sequential(  # Define sequential container for downsampling
            nn.MaxPool2d(2),  # 2x2 max pooling to reduce spatial dimensions
            DoubleConv(in_channels, out_channels)  # Apply DoubleConv after pooling
        )

    def forward(self, x):
        # Forward pass: Apply max pooling and double convolution
        return self.maxpool_conv(x)  # Return downsampled output

class Up(nn.Module):
    # Up: Upsampling block with transposed convolution and DoubleConv
    def __init__(self, in_channels, out_channels):
        # Initialize Up module with input and output channels
        super().__init__()  # Call parent class constructor
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # Transposed convolution for upsampling
        self.conv = DoubleConv(in_channels, out_channels)  # DoubleConv after concatenation

    def forward(self, x1, x2):
        # Forward pass: Upsample x1 and concatenate with x2, then apply DoubleConv
        x1 = self.up(x1)  # Upsample input tensor x1
        diffY = x2.size()[2] - x1.size()[2]  # Compute height difference for padding
        diffX = x2.size()[3] - x1.size()[3]  # Compute width difference for padding
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])  # Pad x1 to match x2 dimensions
        x = torch.cat([x2, x1], dim=1)  # Concatenate x2 and x1 along channel dimension
        return self.conv(x)  # Apply DoubleConv to concatenated tensor

class OutConv(nn.Module):
    # OutConv: Final 1x1 convolution to produce output channels
    def __init__(self, in_channels, out_channels):
        # Initialize OutConv module with input and output channels
        super().__init__()  # Call parent class constructor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 convolution layer

    def forward(self, x):
        # Forward pass: Apply 1x1 convolution
        return self.conv(x)  # Return output of convolution

class UNet(nn.Module):
    # UNet: U-Net architecture for image segmentation
    def __init__(self, n_channels, n_classes):
        # Initialize UNet with number of input channels and output classes
        super(UNet, self).__init__()  # Call parent class constructor
        self.inc = DoubleConv(n_channels, 64)  # Input convolution block
        self.down1 = Down(64, 128)  # First downsampling block
        self.down2 = Down(128, 256)  # Second downsampling block
        self.down3 = Down(256, 512)  # Third downsampling block
        self.down4 = Down(512, 1024)  # Fourth downsampling block
        self.up1 = Up(1024, 512)  # First upsampling block
        self.up2 = Up(512, 256)  # Second upsampling block
        self.up3 = Up(256, 128)  # Third upsampling block
        self.up4 = Up(128, 64)  # Fourth upsampling block
        self.outc = OutConv(64, n_classes)  # Output convolution layer

    def forward(self, x):
        # Forward pass: Implement U-Net encoder-decoder architecture
        x1 = self.inc(x)  # Apply input convolution
        x2 = self.down1(x1)  # First downsampling
        x3 = self.down2(x2)  # Second downsampling
        x4 = self.down3(x3)  # Third downsampling
        x5 = self.down4(x4)  # Fourth downsampling
        x = self.up1(x5, x4)  # First upsampling with skip connection
        x = self.up2(x, x3)  # Second upsampling with skip connection
        x = self.up3(x, x2)  # Third upsampling with skip connection
        x = self.up4(x, x1)  # Fourth upsampling with skip connection
        logits = self.outc(x)  # Apply output convolution to get logits
        return logits  # Return final output

# ------------------ Dataset ------------------
class SegmentationDataset(Dataset):
    # SegmentationDataset: Custom dataset for loading geospatial imagery and masks
    def __init__(self, image_paths, windows, target_size=(256, 256), num_bands=None):
        # Initialize dataset with image paths, windows, target size, and number of bands
        self.image_paths = image_paths  # List of image file paths
        self.windows = windows  # List of raster windows for tiling
        self.target_size = target_size  # Target size for resizing images
        self.num_bands = num_bands  # Number of spectral bands in images

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.windows)  # Number of windows

    def __getitem__(self, idx):
        # Get a sample (image and mask) by index
        img_idx, window = self.windows[idx]  # Get image index and window
        image_path = self.image_paths[img_idx]  # Get corresponding image path

        with rasterio.open(image_path) as src:  # Open raster file
            data = src.read(window=window)  # Read data from specified window
            height, width = data.shape[1], data.shape[2]  # Get height and width of tile
            image = data[:self.num_bands]  # Extract spectral bands
            mask = data[-1]  # Extract mask (last band)

        image = resize(  # Resize image to target size
            image.transpose(1, 2, 0),  # Transpose to (H, W, C) for resizing
            self.target_size + (self.num_bands,),  # Target size with channels
            mode='reflect',  # Use reflect mode for padding
            anti_aliasing=True  # Apply anti-aliasing
        )
        image = image.transpose(2, 0, 1).astype(np.float32)  # Transpose back to (C, H, W) and convert to float32

        for c in range(image.shape[0]):  # Normalize each channel
            channel = image[c]  # Get channel
            min_val, max_val = channel.min(), channel.max()  # Compute min and max values
            if max_val - min_val > 1e-6:  # Avoid division by zero
                image[c] = (channel - min_val) / (max_val - min_val)  # Normalize to [0, 1] 
            else:
                image[c] = 0  # Set to zero if range is too small

        mask = resize(mask, self.target_size, order=0, anti_aliasing=False).astype(np.uint8)  # Resize mask (nearest-neighbor)
        mask = mask[None, :, :]  # Add channel dimension to mask
        mask = (mask > 0).astype(np.uint8)  # Binarize mask

        return torch.from_numpy(image), torch.from_numpy(mask)  # Convert to PyTorch tensors and return

# Function to collect raster files
def get_raster_files(image_base_dir):
    # Collect all .tif files from the specified directory
    return glob.glob(os.path.join(image_base_dir, "**", "*.tif"), recursive=True)  # Return list of TIFF file paths

# Function to compute accuracy
def compute_accuracy(outputs, masks):
    # Compute pixel-wise accuracy for segmentation
    preds = torch.sigmoid(outputs) >= 0.6  # Apply sigmoid and threshold to get predictions
    correct = (preds == masks).float().sum()  # Count correct predictions
    total = masks.numel()  # Total number of pixels
    return correct / total  # Return accuracy

# ------------------ Segmentation Metrics ------------------
def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union (IoU) for binary segmentation"""
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0  # Perfect IoU when both pred and target are empty
    return (intersection / union).item()

def compute_dice_coefficient(pred, target, threshold=0.5):
    """Compute Dice Coefficient for binary segmentation"""
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)
    return dice.item()

def compute_pixel_accuracy(pred, target, threshold=0.5):
    """Compute Pixel Accuracy for binary segmentation"""
    pred = (pred > threshold).float()
    target = target.float()
    
    correct = (pred == target).float().sum()
    total = target.numel()
    return (correct / total).item()

def compute_mae(pred, target):
    """Compute Mean Absolute Error (MAE) for segmentation"""
    pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
    target = target.float()
    
    mae = torch.abs(pred - target).mean()
    return mae.item()

def compute_segmentation_metrics(model, dataloader, device, threshold=0.5):
    """Compute comprehensive segmentation metrics"""
    model.eval()
    ious, dices, pixel_accs, maes = [], [], [], []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Computing segmentation metrics"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            
            # Compute metrics for each sample in the batch
            for i in range(outputs.shape[0]):
                pred = outputs[i].squeeze()
                target = masks[i].squeeze()
                
                iou = compute_iou(pred, target, threshold)
                dice = compute_dice_coefficient(pred, target, threshold)
                pixel_acc = compute_pixel_accuracy(pred, target, threshold)
                mae = compute_mae(pred, target)
                
                ious.append(iou)
                dices.append(dice)
                pixel_accs.append(pixel_acc)
                maes.append(mae)
    
    return {
        'IoU': np.mean(ious),
        'Dice': np.mean(dices),
        'Pixel_Accuracy': np.mean(pixel_accs),
        'MAE': np.mean(maes),
        'IoU_std': np.std(ious),
        'Dice_std': np.std(dices),
        'Pixel_Accuracy_std': np.std(pixel_accs),
        'MAE_std': np.std(maes)
    }

# Function to plot loss and accuracy
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    # Plot training and validation loss and accuracy
    epochs = range(1, len(train_losses) + 1)  # Create epoch range for x-axis
    
    plt.figure(figsize=(12, 5))  # Create figure with specified size
    
    # Plot Loss
    plt.subplot(1, 2, 1)  # Create subplot for loss
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')  # Plot training loss
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')  # Plot validation loss
    plt.title('Training and Validation Loss')  # Set title
    plt.xlabel('Epoch')  # Set x-axis label
    plt.ylabel('Loss')  # Set y-axis label
    plt.legend()  # Show legend
    plt.grid(True)  # Enable grid
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)  # Create subplot for accuracy
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')  # Plot training accuracy
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')  # Plot validation accuracy
    plt.title('Training and Validation Accuracy')  # Set title
    plt.xlabel('Epoch')  # Set x-axis label
    plt.ylabel('Accuracy')  # Set y-axis label
    plt.legend()  # Show legend
    plt.grid(True)  # Enable grid
    
    plt.tight_layout()  # Adjust subplot layout
    
    # Save plot
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Generate timestamp
    loss_acc_fname = f"loss_accuracy_plot_{ts}.png"  # Create filename with timestamp
    save_dir = r'Model_Evaluation'  # Define save directory
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    fname = os.path.join(save_dir, loss_acc_fname)  # Full file path
    plt.savefig(fname, dpi=300, bbox_inches="tight")  # Save plot as PNG
    print(f"Saved loss and accuracy plot → {fname}")  # Print save confirmation
    plt.show()  # Display plot


# Function to compute and save segmentation metrics
def compute_and_save_segmentation_metrics(model, dataloader, device, mode="validation", threshold=0.5):
    """Compute and save segmentation metrics (IoU, Dice, Pixel Accuracy, MAE)"""
    print(f"\nComputing segmentation metrics for {mode} set...")
    
    # Compute metrics
    metrics = compute_segmentation_metrics(model, dataloader, device, threshold)
    
    # Print metrics
    print(f"\n=== Segmentation Metrics ({mode}) ===")
    print(f"IoU (Intersection over Union): {metrics['IoU']:.4f} ± {metrics['IoU_std']:.4f}")
    print(f"Dice Coefficient: {metrics['Dice']:.4f} ± {metrics['Dice_std']:.4f}")
    print(f"Pixel Accuracy: {metrics['Pixel_Accuracy']:.4f} ± {metrics['Pixel_Accuracy_std']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}")
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'Metric': ['IoU', 'Dice Coefficient', 'Pixel Accuracy', 'MAE'],
        'Mean': [metrics['IoU'], metrics['Dice'], metrics['Pixel_Accuracy'], metrics['MAE']],
        'Std': [metrics['IoU_std'], metrics['Dice_std'], metrics['Pixel_Accuracy_std'], metrics['MAE_std']]
    })
    
    # Save metrics to CSV
    metrics_save_dir = r'Model_Evaluation'  # Define save directory
    os.makedirs(metrics_save_dir, exist_ok=True)  # Create directory if it doesn't exist
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Generate timestamp
    fname_metrics = f"segmentation_metrics_{mode}_{ts}.csv"  # Create filename
    metrics_df.to_csv(os.path.join(metrics_save_dir, fname_metrics), index=False)  # Save metrics as CSV
    print(f"Saved segmentation metrics → {os.path.join(metrics_save_dir, fname_metrics)}")  # Print save confirmation
    
    # Create visualization
    plt.figure(figsize=(10, 6))  # Create figure for metrics visualization
    x_pos = np.arange(len(metrics_df))  # X positions for bars
    bars = plt.bar(x_pos, metrics_df['Mean'], yerr=metrics_df['Std'], 
                   capsize=5, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    
    plt.xlabel('Metrics')  # Set x-axis label
    plt.ylabel('Score')  # Set y-axis label
    plt.title(f'Segmentation Metrics ({mode})')  # Set title
    plt.xticks(x_pos, metrics_df['Metric'], rotation=45)  # Set x-axis ticks
    plt.grid(True, alpha=0.3)  # Enable grid
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, metrics_df['Mean'], metrics_df['Std'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()  # Adjust layout
    
    # Save plot
    fname_plot = f"segmentation_metrics_plot_{mode}_{ts}.png"  # Create filename
    plot_path = os.path.join(metrics_save_dir, fname_plot)  # Full file path
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # Save plot as PNG
    print(f"Saved metrics plot → {plot_path}")  # Print save confirmation
    plt.show()  # Display plot
    
    return metrics_df  # Return metrics DataFrame

# ------------------ Main ------------------
def main():
    # Main function: Orchestrates model training, evaluation, and saving
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select GPU if available, else CPU
    image_base_dir = r"Raster_Train"  # Define base directory for raster images

    # System info
    if device.type == "cuda":  # Check if GPU is available
        num_gpus = torch.cuda.device_count()  # Get number of GPUs
        print(f" -> {num_gpus} CUDA device(s) available")  # Print number of GPUs
        for i in range(num_gpus):  # Iterate over GPUs
            print(f"   [GPU {i}] {torch.cuda.get_device_name(i)}")  # Print GPU name
            print(f"       Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")  # Print allocated memory
            print(f"       Memory Cached:    {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")  # Print cached memory
            print(f"       Total Memory:     {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")  # Print total memory
            print(f"       Compute Capability: {torch.cuda.get_device_capability(i)}")  # Print compute capability
    else:  # CPU info
        print(f" -> CPU: {platform.processor() or 'Unknown'}")  # Print CPU name
        print(f" -> CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")  # Print CPU core counts
        print(f" -> RAM available: {psutil.virtual_memory().total / 1024 ** 3:.2f} GB")  # Print available RAM

    # Collect raster files
    image_paths = get_raster_files(image_base_dir)  # Get list of TIFF files
    if not image_paths:  # Check if any files were found
        print("No data")  # Print error message
        return  # Exit function

    # Prepare data
    tile_size = 256  # Define tile size for processing
    all_burn_tiles, all_unburn_tiles = [], []  # Initialize lists for burn and unburn tiles
    max_channels = 0  # Initialize maximum number of channels

    for img_idx, image_path in enumerate(image_paths):  # Iterate over image paths
        with rasterio.open(image_path) as src:  # Open raster file
            height, width = src.shape  # Get image dimensions
            num_bands = src.count - 1  # Get number of bands (excluding mask)
            max_channels = max(max_channels, num_bands)  # Update maximum channels

            windows = [Window(j, i, tile_size, tile_size)  # Create windows for tiling
                       for i in range(0, height, tile_size)  # Iterate over height
                       for j in range(0, width, tile_size)  # Iterate over width
                       if i + tile_size <= height and j + tile_size <= width]  # Ensure window fits

            for w in windows:  # Iterate over windows
                tile_data = src.read(window=w)  # Read tile data
                mask = tile_data[-1]  # Get mask (last band)
                if mask.mean() > 0.1:  # Check if tile is burn (mean mask value > 0.1)
                    all_burn_tiles.append((img_idx, w))  # Add to burn tiles
                else:
                    all_unburn_tiles.append((img_idx, w))  # Add to unburn tiles

    # Balance dataset
    rng = np.random.RandomState(42)  # Initialize random number generator
    n_keep = min(len(all_burn_tiles), len(all_unburn_tiles))  # Get minimum class size
    burn_indices = rng.choice(len(all_burn_tiles), size=n_keep, replace=False)  # Randomly select burn tiles
    unburn_indices = rng.choice(len(all_unburn_tiles), size=n_keep, replace=False)  # Randomly select unburn tiles

    burn_tiles = [all_burn_tiles[i] for i in burn_indices]  # Subset burn tiles
    unburn_tiles = [all_unburn_tiles[i] for i in unburn_indices]  # Subset unburn tiles

    balanced_tiles = burn_tiles + unburn_tiles  # Combine balanced tiles
    tile_labels = [1] * len(burn_tiles) + [0] * len(unburn_tiles)  # Create labels (1 for burn, 0 for unburn)

    # Split into train/validation sets
    train_windows, val_windows, _, _ = train_test_split(  # Split data
        balanced_tiles, tile_labels, test_size=0.2, random_state=42, stratify=tile_labels  # 80-20 split, stratified
    )

    # Build datasets
    train_dataset = SegmentationDataset(image_paths, train_windows, num_bands=max_channels)  # Create training dataset
    val_dataset = SegmentationDataset(image_paths, val_windows, num_bands=max_channels)  # Create validation dataset

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)  # Create training DataLoader
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)  # Create validation DataLoader

    # Loss weighting
    print("Computing pixel class weights...")  # Print status
    burn_pixels, unburn_pixels = 0, 0  # Initialize pixel counts
    for _, masks in train_dataloader:  # Iterate over training data
        burn_pixels += masks.sum().item()  # Count burn pixels
        unburn_pixels += masks.numel() - masks.sum().item()  # Count unburn pixels
    pos_weight = torch.tensor([unburn_pixels / (burn_pixels + 1e-6)], device=device)  # Compute positive weight for loss

    # Model, Loss, Optimizer
    model = UNet(n_channels=max_channels, n_classes=1).to(device)  # Initialize U-Net model and move to device
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Define binary cross-entropy loss with class weighting
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Initialize Adam optimizer with learning rate

    # Training Loop with Accuracy
    num_epochs = 100  # Define number of training epochs
    train_losses, val_losses = [], []  # Initialize lists for losses
    train_accuracies, val_accuracies = [], []  # Initialize lists for accuracies

    for epoch in range(num_epochs):  # Iterate over epochs
        # Training
        model.train()  # Set model to training mode
        running_loss = 0.0  # Initialize running loss
        running_correct = 0  # Initialize running correct predictions
        total_pixels = 0  # Initialize total pixels

        for images, masks in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):  # Iterate over batches
            images, masks = images.to(device), masks.to(device).float()  # Move data to device
            optimizer.zero_grad()  # Clear gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, masks)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item() * images.size(0)  # Update running loss
            
            # Compute accuracy
            correct = compute_accuracy(outputs, masks) * masks.numel()  # Compute correct predictions
            running_correct += correct  # Update running correct
            total_pixels += masks.numel()  # Update total pixels

        epoch_loss = running_loss / len(train_dataloader.dataset)  # Compute average epoch loss
        epoch_acc = running_correct / total_pixels  # Compute epoch accuracy
        train_losses.append(epoch_loss)  # Store training loss
        train_accuracies.append(epoch_acc.item())  # Store training accuracy

        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0  # Initialize validation loss
        val_correct = 0  # Initialize validation correct predictions
        val_pixels = 0  # Initialize validation pixels
        with torch.no_grad():  # Disable gradient computation
            for images, masks in val_dataloader:  # Iterate over validation batches
                images, masks = images.to(device), masks.to(device).float()  # Move data to device
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, masks)  # Compute loss
                val_loss += loss.item() * images.size(0)  # Update validation loss
                correct = compute_accuracy(outputs, masks) * masks.numel()  # Compute correct predictions
                val_correct += correct  # Update validation correct
                val_pixels += masks.numel()  # Update validation pixels

        val_loss = val_loss / len(val_dataloader.dataset)  # Compute average validation loss
        val_acc = val_correct / val_pixels  # Compute validation accuracy
        val_losses.append(val_loss)  # Store validation loss
        val_accuracies.append(val_acc.item())  # Store validation accuracy

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")  # Print epoch metrics

    # Plot loss and accuracy
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)  # Plot training and validation metrics

    # Evaluation with segmentation metrics
    print("\n" + "="*50)
    print("EVALUATION PHASE")
    print("="*50)
    
    # Compute segmentation metrics for validation set
    val_metrics_df = compute_and_save_segmentation_metrics(model, val_dataloader, device, mode="validation", threshold=0.5)
    
    # Compute segmentation metrics for training set (for comparison)
    train_metrics_df = compute_and_save_segmentation_metrics(model, train_dataloader, device, mode="training", threshold=0.5)

    # Save model
    export_dir = "Export_Model"  # Define model save directory
    os.makedirs(export_dir, exist_ok=True)  # Create directory if it doesn't exist
    model_path = os.path.join(export_dir, "unet_wildfire_no_shape_file.pth")  # Define model save path
    torch.save(model.state_dict(), model_path)  # Save model weights
    print(f"✅ Model saved successfully to {model_path}")  # Print save confirmation

if __name__ == "__main__":
    main()  # Run the main function if script is executed directly