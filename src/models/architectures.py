import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics import Accuracy
import yaml
import json
import os

config = yaml.safe_load(open("config.yaml"))

pl.seed_everything(config['data']['seed'])

class ConvNeXtClassifier(pl.LightningModule):
    def __init__(self, num_classes, pretrained=True, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.misclassified_samples = []

        if pretrained:
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        else:
            model = models.convnext_tiny(weights=None)

        # Freeze all layers except the classifier
        for param in model.parameters():
            param.requires_grad = False

        # Modify the classifier for multiclass output
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)

        # Unfreeze the classifier layers
        for param in model.classifier.parameters():
            param.requires_grad = True

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.hparams.learning_rate)

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)

        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('acc/train', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)

        self.log('loss/val', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('acc/val', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.test_acc(preds, labels)

        self.log('loss/test', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('acc/test', acc, on_step=False, on_epoch=True, prog_bar=True)

        # Store misclassified samples
        misclassified_indices = torch.where(preds != labels)[0]
        for idx in misclassified_indices:
          image = inputs[idx].cpu().numpy().tolist() #convert to list to be json serializable
          pred = preds[idx].item()
          label = labels[idx].item()
          self.misclassified_samples.append({
              "image": image,
              "prediction": pred,
              "label": label,
          })

        return loss
    
    def on_test_epoch_end(self):
        # Construct the path to the outputs directory
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of architectures.py
        outputs_dir = os.path.join(current_dir, "..", "..", "outputs")  # Navigate to the outputs directory

        # Ensure the output directory exists
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)

        # Construct the full path to the JSON file
        json_path = os.path.join(outputs_dir, "misclassified_samples.json")

        # Save misclassified samples to the JSON file
        with open(json_path, "w") as f:
            json.dump(self.misclassified_samples, f, indent=4)
        
        print(f"Misclassified samples saved to {json_path}")

        self.misclassified_samples.clear()

    def configure_optimizers(self):
        return self.optimizer

class ResNet50Classifier(pl.LightningModule):
    def __init__(self, num_classes, pretrained=True, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(weights=None)

        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the final fully connected layer
        for param in model.fc.parameters():
            param.requires_grad = True

        # Modify the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.hparams.learning_rate)

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # Log metrics
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)

        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('acc/train', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        # print(f"Val - Labels: {labels}")
        # print(f"Val - Predictions: {preds}")
        # print(f"Val - Outputs: {outputs}")

        acc = self.val_acc(preds, labels)
        # self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('loss/val', loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('acc/val', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # Log metrics
        preds = torch.argmax(outputs, dim=1)
        acc = self.test_acc(preds, labels)

        # self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('loss/test', loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('acc/test', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return self.optimizer


class Scratch(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        size = 128
        self.conv1_1 = nn.Conv2d(3, size, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(size, size, 3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(size, size, 3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(size, size*2, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(size*2, size*2, 3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(size*2, size*2, 3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(size*2, size*4, 3, stride=1, padding=1)
        
        self.pool  = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout(p = 0.3)
        self.dropout2 = nn.Dropout(p = 0.3)
        
        # 32 -> 16 -> 8
        self.fc1     = nn.Linear(size*4 * 16 * 16, size)
        self.fc2     = nn.Linear(size, size)
        self.fc3     = nn.Linear(size , num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.leaky_relu(self.conv1_1(x))
        x = F.leaky_relu(self.conv1_2(x))
        x = F.leaky_relu(self.conv1_3(x))

        x = self.pool(x)
        x = F.leaky_relu(self.conv2_1(x))
        x = F.leaky_relu(self.conv2_2(x))
        x = F.leaky_relu(self.conv2_3(x))

        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1) # Flatten

        x=F.leaky_relu(self.fc1(x))
        x=self.dropout1(x)
        x=F.leaky_relu(self.fc2(x))
        x=self.dropout2(x)
        x=self.fc3(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.accuracy(outputs, labels)
        # self.log('train_acc', self.accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('acc/train', self.accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        # self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('loss/val', loss, on_epoch=True, prog_bar=True)
        self.accuracy(outputs, labels)
        # self.log('val_acc', self.accuracy, on_epoch=True, prog_bar=True)
        self.log('acc/val', self.accuracy, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        # self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('loss/test', loss, on_epoch=True, prog_bar=True)
        self.accuracy(outputs, labels)
        # self.log('test_acc', self.accuracy, on_epoch=True, prog_bar=True)
        self.log('acc/test', self.accuracy, on_epoch=True, prog_bar=True)
        return loss

if __name__ == "__main__":
    # Example usage
    num_classes = 2
    model = ResNet50Classifier(num_classes)
    print(model)

    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print("Output shape:", output.shape)