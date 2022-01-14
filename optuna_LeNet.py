"""
optuna MNIST fully connected network - https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
turned into CNN, starting with close architecture to LeNet - https://en.wikipedia.org/wiki/Convolutional_neural_network#/media/File:Comparison_image_neural_networks.svg
* slight change, in first convolutional layer. No padding
train and evaluate loop inspired from - https://www.kaggle.com/sdelecourt/cnn-with-pytorch-for-mnist
""" 

### test of optimizing LeNet with optuna ###

# imports

import torch
import torch.nn as nn

import torchvision
from torchvision import datasets
from torchvision import transforms

import optuna
from optuna.trial import TrialState

import os


### Parameters ###

# architecture params
initial_height_width = 28 # 28x28
ks = 5 # conv2d kernel size
pad = 2 
classes = 10

# training params
num_epochs = 1
batch_size = 50
n_train_examples = batch_size * 30
n_test_examples = batch_size * 10
# learning_rate = 0.1


### Import MNIST dataset ###

def get_mnist():
    
    DIR = os.getcwd()

    # if MNIST not found in directory, will download to directory
    dl = False
    if (not os.path.isdir("MNIST")):
        dl = True
    
    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(root=DIR, 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=dl), 
                                    batch_size=batch_size, 
                                    shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(root=DIR, 
                                            train=False, 
                                            transform=transforms.ToTensor(),
                                            download=dl), 
                                    batch_size=batch_size, 
                                    shuffle=True) 

    return train_loader, test_loader  


### variable model ###

def define_model(trial):
    # starting with layer number and architecture numbers
    
    cnn_n_layers = 2 # trial.suggest_int("n_layers", 1, 3) # for now layers fixed
    ff_n_layers = 2 # trial.suggest_int("n_layers", 1, 3) # fixed
    layers = []

    in_channels = 1
    height_width = initial_height_width

    for i in range(cnn_n_layers):
        out_channels = trial.suggest_int("n_units_convolution_l{}".format(i), 1, 100)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=ks))
        layers.append(nn.MaxPool2d(kernel_size=pad))
        layers.append(nn.ReLU())
        in_channels = out_channels
        height_width = int((height_width - (ks-1)) / pad)
    
    layers.append(nn.Flatten())
    in_features = in_channels * (height_width**2)

    for j in range(ff_n_layers):
        out_features = trial.suggest_int("n_units_linear_l{}".format(j), 1, 100)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features

    layers.append(nn.Linear(in_features, classes))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


### training and testing of accuracy ###

# Get the MNIST dataset.
train_loader, valid_loader = get_mnist()

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def objective(trial):

    # model initialization
    net = define_model(trial).to(device)

    # optimizer and error function
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    error = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(num_epochs):
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * batch_size >= n_train_examples:
                break

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = error(output, target)
            loss.backward()
            optimizer.step()


        # testing of model within epoch for evaluation of a run that has "pruned"
        net.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * batch_size >= n_test_examples:
                    break
                data, target = data.to(device), target.to(device)
                output = net(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), n_test_examples)

        # checkes if arcitecture is worth continueing, note this is in the epoch loop
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value. (report arguments are intermediate values)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
