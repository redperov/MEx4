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
        self.bn0 = nn.BatchNorm1d(100)
        self.bn1 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        return F.log_softmax(self.fc2(x), dim=1)


class ModelD(nn.Module):
    """
    Convolutional neural network.
    """
    def __init__(self):
        super(ModelD, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def main():

    # Load the data.
    train_loader, test_loader = load_data()

    # Split the examples to training and validation sets 80:20.
    train_loader, validation_loader = split_data(train_loader.dataset)

    # Initialize models.
    model_a = ModelA(image_size=28 * 28)
    model_b = ModelB(image_size=28 * 28)
    model_c = ModelC(image_size=28 * 28)
    model_d = ModelD()

    # Initialize hyper-parameters.
    lr = 0.05
    epochs = 10

    # Setting the optimizers.
    optimizer_a = optim.SGD(model_a.parameters(), lr=lr)
    optimizer_b = optim.SGD(model_b.parameters(), lr=lr)
    optimizer_c = optim.SGD(model_c.parameters(), lr=lr)
    optimizer_d = optim.SGD(model_d.parameters(), lr=lr)

    # Train and plot model A
    train_and_plot(model_a, optimizer_a, epochs, train_loader, validation_loader, test_loader)

    # Train and plot model B
    train_and_plot(model_b, optimizer_b, epochs, train_loader, validation_loader, test_loader)

    # Train and plot model C
    train_and_plot(model_c, optimizer_c, epochs, train_loader, validation_loader, test_loader)

    # Train and plot model C
    train_and_plot(model_d, optimizer_d, epochs, train_loader, validation_loader, test_loader)

    # Write the prediction of the best model to a file.
    write_prediction(model_d, test_loader)


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
        batch_size=64, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, download=True,
                              transform=t),
        batch_size=64, shuffle=False)

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
                    batch_size=64, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(data_set,
                    batch_size=64, sampler=validation_sampler)

    return train_loader, validation_loader


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
        train(model, optimizer, train_loader)

        # Get training loss.
        train_loss = test(epoch, model, train_loader, "Training set")
        train_loss_list.append(train_loss)

        # Get validation loss.
        validation_loss = test(epoch, model, validation_loader, "Validation set")
        validation_loss_list.append(validation_loss)

    # Test the model.
    test(0, model, test_loader, "Test set")

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


def train(model, optimizer, train_loader):
    """
    Trains the network.
    :param model: neural network
    :param optimizer: optimizer
    :param train_loader: train loader
    :return: training average loss
    """
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def test(epoch, model, test_loader, set_type):
    """
    Tests the final model.
    :param model: neural network
    :param test_loader: test loader
    :return: test loss
    """
    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        counter += 1

    test_loss /= len(test_loader.sampler)
    print('\n{}: Epoch: {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        set_type, epoch, test_loss, correct, len(test_loader.sampler),
        100.0 * correct / len(test_loader.sampler)))

    return test_loss


def write_prediction(model, test_loader):
    """
    Performs a prediction over the test set and writes it to a file.
    :param model: model
    :param test_loader: test loader
    :return: None
    """
    with open("test.pred", "w") as test_file:
        for data, target in test_loader:

            # Pass the example through the classifier.
            output = model(data)

            # Extract the predicted label.
            predicted_value = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

            for value in predicted_value:
                test_file.write(str(value.item()) + '\n')


if __name__ == "__main__":
    main()
