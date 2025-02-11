import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.network import AttentiveCNN
from utils.network import TransformerEncoderDecoder
from utils.dataloader import TimeSeriesDataset

# Loss Function
class RTreeLoss(nn.Module):
    def __init__(self, lambda_iou=1.0, lambda_type=1.0):
        super(RTreeLoss, self).__init__()
        self.lambda_iou = lambda_iou
        self.lambda_type = lambda_type

    def forward(self, pred_iou, true_iou, pred_type, true_type):
        iou_loss = F.mse_loss(pred_iou, true_iou)
        type_loss = F.binary_cross_entropy_with_logits(pred_type, true_type)
        total_loss = self.lambda_iou * iou_loss + self.lambda_type * type_loss
        return total_loss

def train_model():
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 20
    input_dim = 256  # Feature dimension from AttentiveCNN
    model_dim = 512
    num_heads = 8
    num_layers = 6
    output_dim = 256  # Dimension of the predicted R*-tree sequence

    # Data Preparation
    train_dataset = TimeSeriesDataset("dataprocess/data/processed/splitByGlobalTime")
    val_dataset = TimeSeriesDataset("dataprocess/data/processed/splitByGlobalTime")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model Initialization
    cnn_model = AttentiveCNN(input_channels=3, output_channels=input_dim)
    transformer_model = TransformerEncoderDecoder(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers, output_dim=output_dim)
    criterion = RTreeLoss()
    optimizer = optim.Adam(list(cnn_model.parameters()) + list(transformer_model.parameters()), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)
    transformer_model.to(device)

    # Training Loop
    for epoch in range(num_epochs):
        cnn_model.train()
        transformer_model.train()
        total_loss = 0

        for batch in train_loader:
            # Load batch data
            images, target_sequences = batch
            images = images.to(device)
            target_sequences = target_sequences.to(device)

            # Forward pass
            spatial_features = cnn_model(images)
            src = spatial_features.permute(1, 0, 2)  # Transformer expects (seq_len, batch, feature)
            tgt_input = target_sequences[:, :-1, :].permute(1, 0, 2)  # Remove last token for decoder input
            tgt_output = target_sequences[:, 1:, :].permute(1, 0, 2)  # Shift target for loss computation

            outputs = transformer_model(src, tgt_input)

            # Compute loss
            pred_iou, true_iou = outputs[:, :, :1], tgt_output[:, :, :1]  # Assuming IoU is in the first dimension
            pred_type, true_type = outputs[:, :, 1:], tgt_output[:, :, 1:]  # Assuming type is in the remaining dimensions
            loss = criterion(pred_iou, true_iou, pred_type, true_type)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

        # Validation Loop (Optional)
        validate_model(cnn_model, transformer_model, val_loader, criterion, device)

    # Save model
    torch.save({
        'cnn_model': cnn_model.state_dict(),
        'transformer_model': transformer_model.state_dict()
    }, "trained_model.pth")

def validate_model(cnn_model, transformer_model, val_loader, criterion, device):
    cnn_model.eval()
    transformer_model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            images, target_sequences = batch
            images = images.to(device)
            target_sequences = target_sequences.to(device)

            spatial_features = cnn_model(images)
            src = spatial_features.permute(1, 0, 2)
            tgt_input = target_sequences[:, :-1, :].permute(1, 0, 2)
            tgt_output = target_sequences[:, 1:, :].permute(1, 0, 2)

            outputs = transformer_model(src, tgt_input)
            pred_iou, true_iou = outputs[:, :, :1], tgt_output[:, :, :1]
            pred_type, true_type = outputs[:, :, 1:], tgt_output[:, :, 1:]

            val_loss = criterion(pred_iou, true_iou, pred_type, true_type)
            total_val_loss += val_loss.item()

    print(f"Validation Loss: {total_val_loss / len(val_loader):.4f}")

if __name__ == "__main__":
    train_model()
