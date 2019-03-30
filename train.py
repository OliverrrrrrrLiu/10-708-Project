import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import resnet

parser = argparse.ArgumentParser(description = "PyTorch model training")
parser.add_argument("--epochs", default = 100, type = int, help = "Number of training epochs")
parser.add_argument("--batch_size", default = 64, type = int, help = "Size of mini-batch")
parser.add_argument("--learning_rate", default = 1e-3, type = float, help = "Initial learning rate")
#parser.add_argument("--momentum", default = 0.9, type = float, help = "Momentum)
parser.add_argument("--weight_decay", default = 1e-4, type = float, help = "Weight decay")
parser.add_argument("--dropout_prob", default = 0.0, type = float, help = "Droptout probability")
parser.add_argument("--experiment_id", default = "ResNet-18", type = str, help = "Name of experiment")
parser.add_argument("--sdr", default = False, help = "Whether to use the stochastic delta rule")
parser.add_argument("--beta", default = 5.0, type = float, help = "SDR beta value")
parser.add_argument("--zeta", default = 0.99, type = float, help = "SDR zeta value")
parser.add_argument("--zeta_drop", default = 1, type = int, help = "Control rate of zeta drop") # what is this?
parser.add_argument("--file_path", default = False, help = "Path to save parameter weights and standard deviations")
parser.add_argument("--std_init", default = "xavier", type = str, help = "Initialization method to standard deviations")
parser.add_argument("--print_freq", default = 10, type = int, help = "Frequency to print training statistics")

# dataset
def load_CIFAR():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    valset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    train_loader = data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)
    val_loader  = data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=8)
    return train_loader, val_loader

class AverageMeter(object):
    def __init__(self, alpha = 0.9):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        if self.alpha is None:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        else:
            self.avg = self.alpha * val + (1 - self.alpha) * val

def accuracy(outputs, labels):
    correct, total = 0, 0
    with torch.no_grad():
        batch_size = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
    return correct, batch_size

def init_std_halved_xavier(parameter):
    lower = 0.0
    upper = np.sqrt(2 / np.product(parameter.shape)) * 0.5
    std_init = torch.randn(parameter.shape)
    max_val = torch.max(std_init)
    min_val = torch.min(std_init)
    std_init = ((upper - lower) / (max_val - min_val)).float() * (std_init - min_val)
    return std_init

def main():
    args = parser.parse_args()
    # data loading and model construction
    print("Loading CIFAR-10 datasets...")
    train_loader, val_loader = load_CIFAR()
    print("Loading completed!")
    model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=10)
    model.cuda()
    dropout_prob = 0.0 if args.sdr else args.dropout_prob
    if args.sdr:
        model.sdr = args.sdr
        model.beta = args.beta
        model.zeta = args.zeta
        model.zeta_init = args.zeta
        model.zeta_drop = args.zeta_drop
        model.std_lst = []
    else:
        model.sdr = False

    if args.file_path:
        init_weights = [np.asarray(p.data.cpu()) for p in model.parameters()]
        fname_w = os.path.join(args.file_path, "weight", "init_weights.pt")
        torch.save(model.state_dict(), fname_w)
        del init_weights

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    if args.std_init == "xavier":
        std_init_fn = init_std_halved_xavier

    train_losses, train_accuracies, val_accuracies = [], [], []
    for e in range(args.epochs):
        print("Training on epoch {}...".format(e+1))
        train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, e, std_init_fn)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print("Evaluating on validation set...")
        val_acc = validate(val_loader, model, criterion, e)
        print("Evaluation completed!")
        val_accuracies.append(val_acc)

def train(args, train_loader, model, criterion, optimizer, curr_epoch, std_init_fn = init_std_halved_xavier):
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        if model.sdr:
            if curr_epoch == 0 and i == 0:
                for param in model.parameters():
                    std = std_init_fn(param) # type torch.tensor
                    model.std_lst.append(std)
                if args.file_path:
                    fname_std = os.path.join(args.file_path, "std", "init_std.npy")
                    std_lst_ = [np.asarray(std) for std in model.std_lst]
                    np.save(fname_std, std_lst_)
                    del std_lst_
            elif i in [args.batch_size // 2 - 1, args.batch_size - 1]:
                for j, param in enumerate(model.parameters()):
                    model.std_lst[j] = model.zeta * (torch.abs(model.beta * param.grad) + model.std_lst[j].cuda())

            store = []
            for j, param in enumerate(model.parameters()):
                store.append(param.data)
                param.data = torch.distributions.Normal(param.data, model.std_lst[j].cuda()).sample()
        outputs = model(inputs)
        if model.sdr:
            for sampled_param, stored_param in zip(model.parameters(), store):
                sampled_param.data = stored_param
        loss = criterion(outputs, labels)
        correct, batch_size  = accuracy(outputs, labels)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(100.0 * correct / batch_size, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del inputs
        del labels
        del store
        if i % args.print_freq == 0:
            print('[%d, %5d] loss: %.3f, accuracy: %.3f' % (curr_epoch+1, i+1, losses.avg, accuracies.avg))
    if model.sdr and args.file_path:
        std_lst_ = [np.asarray(std.cpu()) for std in model.std_lst]
        fname_std = os.path.join(args.file_path, "std", "std_epoch_{}.npy".format(curr_epoch))
        np.save(fname_std, std_lst_)
        fname_w = os.path.join(args.file_path, "weight", "model_epoch_{}.pt".format(curr_epoch))
        torch.save(model.state_dict, fname_w)
        del std_lst_
    return losses.avg, accuracies.avg

def validate(val_loader, model, criterion, curr_epoch):
    correct, total = 0, 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_correct, batch_size = accuracy(outputs, labels)
            correct += batch_correct
            total += batch_size
            del inputs
            del labels
    res = 100.0 * correct / total
    print("Classification accuracy on the validation set is {:.3f}".format(res))
    return res

if __name__ == "__main__":
    main()
