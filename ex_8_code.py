import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt


class ModelA(nn.Module):
    """
    Neural network with two hidden layers.
    """
    def __init__(self, image_size):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class ModelB(nn.Module):
    """
    Neural network with two hidden layers and Dropout.
    """
    def __init__(self, image_size):
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.drop1 = nn.Dropout()
        self.fc1 = nn.Linear(100, 50)
        self.drop2 = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.log_softmax(self.fc2(x), dim=1)


class ModelC(nn.Module):
    """
    Neural network with two hidden layers and Batch Normalization.
    """
    def __init__(self, image_size):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.bn1(x)
        x = F.relu(self.fc0(x))
        x = self.bn2(x)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


def main():

    # Load the data.
    train_loader, test_loader = load_data()

    # Split the examples to training and validation sets 80:20.
    train_loader, validation_loader = split_data(train_loader.dataset)

    # Initialize models.
    model_a = ModelA(image_size=28 * 28)
    model_b = ModelB(image_size=28 * 28)
    model_c = ModelC(image_size=28 * 28)

    # Initialize hyper-parameters.
    lr = 0.01
    epochs = 10

    # Setting the optimizers.
    optimizer_a = optim.SGD(model_a.parameters(), lr=lr)
    optimizer_b = optim.SGD(model_b.parameters(), lr=lr)
    optimizer_c = optim.SGD(model_c.parameters(), lr=lr)

    # Train and plot model A
    # train_and_plot(model_a, optimizer_a, epochs, train_loader, validation_loader, test_loader)

    # Train and plot model B
    train_and_plot(model_b, optimizer_b, epochs, train_loader, validation_loader, test_loader)

    # Train and plot model C
    train_and_plot(model_c, optimizer_c, epochs, train_loader, validation_loader, test_loader)


def train_and_plot(model, optimizer, epochs, train_loader, validation_loader, test_loader):
    """
    Trains and plots the results of a given model.
    :param model: model
    :param optimizer: optimizer
    :param epochs: number of epochs
    :param train_loader: train loader
    :param validation_loader: validation loader
    :param test_loader: test loader
    :return: None
    """

    train_loss_list = []
    validation_loss_list = []

    for epoch in range(epochs):
        # Train the model.
        train_loss = train(epoch, model, optimizer, train_loader)
        train_loss_list.append(train_loss)

        # Validate the model.
        validation_loss = validate(epoch, model, validation_loader)
        validation_loss_list.append(validation_loss)

    # Test the model.
    test(model, test_loader)

    # Plot average training and validation loss VS number of epochs
    plot_graph(epochs, train_loss_list, validation_loss_list)


def plot_graph(epochs, train_loss_list, validation_loss_list):
    """
    Plots a graph of training and validation average loss VS number of epochs.
    :param epochs: number of epochs
    :param train_loss_list: train loss list
    :param validation_loss_list: validation loss list
    :return: None
    """
    epochs_list = list(range(epochs))
    plt.plot(epochs_list, train_loss_list, 'b', label="training loss")
    plt.plot(epochs_list, validation_loss_list, 'r--', label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Average loss")
    plt.legend()
    plt.show()


def load_data():
    """
    Loads the training and testing data.
    :return: train_loader, test_loader
    """
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True,
                       transform=t),
        batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=t),
        batch_size=64, shuffle=True)

    return train_loader, test_loader


def split_data(data_set):
    """
    Splits the training data to training and validation 80:20.
    :param data_set: data_set
    :return: train_loader, validation_loader
    """
    num_train = len(data_set)
    indices = list(range(num_train))
    split = int(num_train * 0.2)

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # Perform random split using subset random samples.
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(data_set,
                    batch_size=1, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(data_set,
                    batch_size=1, sampler=validation_sampler)

    return train_loader, validation_loader


def train(epoch, model, optimizer, train_loader):
    """
    Trains the network.
    :param epoch: number of epoch
    :param model: neural network
    :param optimizer: optimizer
    :param train_loader: train loader
    :return: training average loss
    """
    model.train()
    training_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)

        # Calculation for checking the loss and accuracy.
        training_loss += loss.item() #F.nll_loss(output, labels, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

        loss.backward()
        optimizer.step()

    training_loss /= len(train_loader)
    training_accuracy = 100.0 * correct / len(train_loader)
    print('\nTraining set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, training_loss, correct, len(train_loader),
        training_accuracy))

    return training_loss


def validate(epoch, model, validation_loader):
    """
    Performs a validation check on the current model.
    :param epoch: number of epoch
    :param model: neural network
    :param validation_loader: validation loader
    :return: Validation average loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in validation_loader:
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(validation_loader)
    validation_accuracy = 100.0 * correct / len(validation_loader)
    print('Validation set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, validation_loss, correct, len(validation_loader),
        validation_accuracy))

    return validation_loss


def test(model, test_loader):
    """
    Tests the final model.
    :param model: neural network
    :param test_loader: test loader
    :return: None
    """
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    main()
