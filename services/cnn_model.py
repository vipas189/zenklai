import torch
from torch import nn
from torch.utils.data import DataLoader
from services.image_processing import image_processing
from services.early_stop import early_stop
from models.simplecnn import SimpleCNN
from models.mydataset import MyDataset
import os
from PIL import Image
from torchvision import transforms

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def start(
    X_train, X_val, y_train, y_val, label_to_index, index_to_label, hyperparameters
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_scores": [],
        "epoch_train_loss": 0,
        "epoch_train_acc": 0,
        "running_train_loss": 0,  # Accumulate total loss for all individual images in an epoch
        "correct_train": 0,
        "total_train": 0,  # Count total individual images processed in an epoch
        "epoch_val_loss": 0,
        "epoch_val_acc": 0,
        "running_val_loss": 0,  # Accumulate total loss for all individual images in an epoch
        "correct_val": 0,
        "total_val": 0,  # Count total individual images processed in an epoch
        "best_val_accuracy": 0,
        "best_val_loss": float("inf"),
        "not_improved_val_acc": 0,
        "best_model_wts": None,
        "index_to_label_mapping": index_to_label,
    }
    train_data = MyDataset(X_train, y_train)
    val_data = MyDataset(X_val, y_val, validation=True)
    train_loader = DataLoader(
        train_data, batch_size=hyperparameters.get("batch_size"), shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=hyperparameters.get("batch_size"), shuffle=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(label_to_index)).to(device)

    opt_name = hyperparameters.get("optimizer", "adam").lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparameters.get("learning_rate"),
            weight_decay=hyperparameters.get("weight_decay"),
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hyperparameters.get("learning_rate"),
            weight_decay=hyperparameters.get("weight_decay"),
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=hyperparameters.get("learning_rate"),
            weight_decay=hyperparameters.get("weight_decay"),
            momentum=hyperparameters.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    loss_fn = nn.CrossEntropyLoss()

    print("Starting Training...")
    epochs = hyperparameters.get("epochs")
    for epoch in range(epochs):
        train(model, loss_fn, optimizer, train_loader, history, device, hyperparameters)
        val(model, loss_fn, val_loader, history, device, hyperparameters)
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {history.get('epoch_train_loss'):.4f} | Train Acc: {history.get('epoch_train_acc'):.4f} | "
            f"Val Loss: {history.get('epoch_val_loss'):.4f} | Val Acc: {history.get('epoch_val_acc'):.4f} - Best: {history.get('best_val_accuracy'):.4f}"
        )
        if history.get("epoch_val_acc") > history.get("best_val_accuracy"):
            torch.save(model, f"best-models/cnn_{history.get('epoch_val_acc')}.pth")
        if early_stop(model, history, hyperparameters):
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("Training finished.")


def train(model, loss_fn, optimizer, train_loader, history, device, hyperparameters):
    history["running_train_loss"] = 0.0
    history["correct_train"] = 0
    history["total_train"] = 0
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        history["running_train_loss"] += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs, 1)
        history["total_train"] += labels.size(0)
        history["correct_train"] += (predicted == labels).sum().item()

    if history["total_train"] > 0:
        history["epoch_train_loss"] = history.get("running_train_loss") / history.get(
            "total_train"
        )
        history["epoch_train_acc"] = history.get("correct_train") / history.get(
            "total_train"
        )
    else:
        history["epoch_train_loss"] = 0.0
        history["epoch_train_acc"] = 0.0

    history["train_loss"].append(history.get("epoch_train_loss"))
    history["train_acc"].append(history.get("epoch_train_acc"))


def val(model, loss_fn, val_loader, history, device, hyperparameters):
    history["running_val_loss"] = 0.0
    history["correct_val"] = 0
    history["total_val"] = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            history["running_val_loss"] += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            history["total_val"] += labels.size(0)
            history["correct_val"] += (predicted == labels).sum().item()

    # Calculate epoch metrics based on the total accumulated values
    # epoch_val_loss is the average loss per individual image across the epoch
    if history["total_val"] > 0:
        history["epoch_val_loss"] = history.get("running_val_loss") / history.get(
            "total_val"
        )
        history["epoch_val_acc"] = history.get("correct_val") / history.get("total_val")
    else:
        history["epoch_val_loss"] = 0.0
        history["epoch_val_acc"] = 0.0

    history["val_loss"].append(history.get("epoch_val_loss"))
    history["val_acc"].append(history.get("epoch_val_acc"))


def run_test():
    class_map = {
        0: "Speed limit (20km/h)",
        1: "Speed limit (30km/h)",
        2: "Speed limit (50km/h)",
        3: "Speed limit (60km/h)",
        4: "Speed limit (70km/h)",
        5: "Speed limit (80km/h)",
        6: "End of speed limit (80km/h)",
        7: "Speed limit (100km/h)",
        8: "Speed limit (120km/h)",
        9: "No passing",
        10: "No passing for vehicles over 3.5 metric tons",
        11: "Right-of-way at the next intersection",
        12: "Priority road",
        13: "Yield",
        14: "Stop",
        15: "No vehicles",
        16: "Vehicles over 3.5 metric tons prohibited",
        17: "No entry",
        18: "General caution",
        19: "Dangerous curve to the left",
        20: "Dangerous curve to the right",
        21: "Double curve",
        22: "Bumpy road",
        23: "Slippery road",
        24: "Road narrows on the right",
        25: "Road work",
        26: "Traffic signals",
        27: "Pedestrians",
        28: "Children crossing",
        29: "Bicycles crossing",
        30: "Beware of ice/snow",
        31: "Wild animals crossing",
        32: "End of all speed and passing limits",
        33: "Turn right ahead",
        34: "Turn left ahead",
        35: "Ahead only",
        36: "Go straight or right",
        37: "Go straight or left",
        38: "Keep right",
        39: "Keep left",
        40: "Roundabout mandatory",
        41: "End of no passing",
        42: "End of no passing by vehicles over 3.5 metric tons",
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("best-models/cnn_0.9622747747747747.pth", weights_only=False).to(
        device
    )
    model.eval()

    img_path = "static/uploads/test-images/image_name.png"
    img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),  # adjust to your model input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # standard for many pretrained models
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    input_tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = class_map[predicted.item()]

    return label
