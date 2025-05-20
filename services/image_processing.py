import torchvision.transforms as transforms


def image_processing(img, data, validation=False):
    x1, y1, x2, y2 = data[0], data[1], data[2], data[3]
    roi = img.crop((x1, y1, x2, y2))
    train_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if validation:
        return val_transform(roi)
    return train_transform(roi)
