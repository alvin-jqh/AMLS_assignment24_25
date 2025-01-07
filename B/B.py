import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import numpy as np
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

import medmnist
from medmnist import INFO, Evaluator, BloodMNIST

cudnn.benchmark = True

# BATCH_SIZE = 8
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class modelbb():
    """
    This is a class that handles the training and testing of the CNN model used for task B
    """
    def __init__(self, BATCH_SIZE, load_model):
        self._BATCH_SIZE = BATCH_SIZE
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # prepare the data into dataloaders
        self.dataloaders, self.dataset_sizes, self.class_names, self.test_loader = self.fetch_data()

        # set up as no model, later ask user to choose whether they want to load or train new model
        self.model = self.setup_model(load_model)
        if self.model is not None:
            self.model.to(self.device)

    def fetch_data(self):
        """
        Returns:
            dataloaders: a dictionary of two dataloaders, "train" and "val" dataloaders in batchsize for bloodmnist
            dataset_size: number of data points in each split
            class_names: dictionary keys is the encoded value and the value is the name of the class
        """
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomResizedCrop(224),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                # transforms.RandomResizedCrop(224),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # get the path for the datasets
        dir = os.getcwd()

        data_flag = "bloodmnist"
        data_dir = "Datasets"

        PATH = os.path.join(dir, data_dir)

        image_datasets = {x: BloodMNIST(split= x, transform= data_transforms[x], download=False, root=PATH)
                    for x in ["train", "val"]} 

        dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=self._BATCH_SIZE, shuffle=True)
                    for x in ['train', 'val']}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        test_dataset = BloodMNIST(split= "test", transform= data_transforms["val"], download=False, root=PATH)

        test_loader = data.DataLoader(test_dataset, batch_size=self._BATCH_SIZE, shuffle = False)

        info = INFO[data_flag]
        class_names = info["label"]
        class_names = {int(key): value for key, value in class_names.items()}

        return dataloaders, dataset_sizes, class_names, test_loader

    def setup_model(self, load=True):
        """
        Loads a saved model or a new model
        Args:
            Load: True by default, if True, loads the model from a saved file, if False, load a pytorch model

        Returns:
            Model: A new ResNet if load = False, A model that has already been trained

        """
        # Load from a model that is already saved
        if load:
            file_path = os.path.join(os.getcwd(), "B", "cnn_model_weights.pt")
            model = torch.load(file_path, weights_only = False)
            print(f"Model Loaded from {file_path}")
        else:
            # load resnet from 
            model = models.resnet18(weights="IMAGENET1K_V1")

            # replaces the fc layer at the end to match the number of classes
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names))
            print("New ResNet loaded")
        return model
    
    def imshow(self, inp, title=None):
        """Display image for Tensor."""
        # change from (C,H,W) to (H, W, C)
        inp = inp.numpy().transpose((1, 2, 0))
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def display_images(self):
        inputs, classes = next(iter(self.dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        # plot some images
        self.imshow(out, title=[self.class_names[int(x)] for x in classes])
        plt.show()

    def train_model(self, criterion, optimizer, scheduler, num_epochs=25):
        print("Training starting...")
        since = time.time()

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(self.model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in self.dataloaders[phase]:
                        labels = labels.squeeze()
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(self.model.state_dict(), best_model_params_path)

                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

            # load best model weights
            self.model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
        # return model

    def test_accuracy(self):
        self.model.eval()
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                labels = labels.squeeze()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)
        
        accuracy = running_corrects.double()/total_samples
        print(f"test accuracy: {accuracy}")

    def visualize_model(self, num_images=6):
        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {self.class_names[preds[j].item()]}\n actual: {self.class_names[labels[j].item()]}')
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        # plt.show(block=False)
                        self.model.train(mode=was_training)
                        return
            self.model.train(mode=was_training)



def load_test():
    bb = modelbb(BATCH_SIZE=32, load_model=True)

    # criterion = nn.CrossEntropyLoss()

    # # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(bb.model.parameters(), lr=0.001, momentum=0.9)

    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # bb.train_model(criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler)
    bb.visualize_model()
    bb.test_accuracy()
    plt.show()


if __name__ == "__main__":
    load_test()