from torchvision import transforms


data_transform = {
    "train":
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    "val":
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    "test":
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    "attack":
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
}
