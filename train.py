import argparse
from torchvision import models
from funcs import load_data, build_model, train_model, save_model, test_model

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action = 'store')

parser.add_argument('--arch', action='store', default = 'densenet121')

parser.add_argument('--save_dir', action = 'store', default = 'mycheckpoint.pth')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=float, default = 0.006)

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'h_units', type=int, default = 512)

parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 2)

parser.add_argument('--gpu', action = "store_true", default = False)

results = parser.parse_args()

data_dir_in = results.data_dir
arch_in = results.arch
save_dir_in = results.save_dir
lr_in = results.lr
hidden_size = results.h_units
epochs = results.num_epochs
gpu_on = results.gpu

train_data, trainloader, validloader, testloader = load_data(data_dir_in) 

model = getattr(models, arch_in) (pretrained = True)
for param in model.parameters():
    param.requires_grad = False

build_model(model, hidden_size)

train_model(model, gpu_on, trainloader, validloader, lr_in, epochs)

test_model(model, gpu_on, testloader)

save_model(model, arch_in, train_data, save_dir_in)
