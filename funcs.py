import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np

def load_data(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms= transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle = True)
    
    return train_data, trainloader, validloader, testloader

def build_model(model, hidden_size):
    input_size = model.classifier.in_features
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_size)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_size, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    
def train_model(model, gpu, trainloader, validloader, learn_rate, epochs):
    device = torch.device("cuda" if gpu == True else "cpu")
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    steps = 0
    running_loss = 0
    print_every = 10
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

def save_model(model, arch_in, train_data, filename):
    model.to('cpu')
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'mapping': model.class_to_idx,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'model_name': arch_in}
    torch.save(checkpoint, filename)
    
def test_model(model, gpu, testloader):
    device = torch.device("cuda" if gpu == True else "cpu")
    model.to(device)
    accuracy = 0 

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(testloader):.3f}")
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['model_name']) (pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']
            
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    ins = Image.open(image)
    
    width = ins.width
    height = ins.height
    
    if width > height:
        width = int(width / height * 256)
        height = 256
    else:
        height = int(height / width * 256)
        width = 256
    ins.thumbnail((width, height),Image.ANTIALIAS)
    
    left = (width - 224) / 2
    upper = (height - 224) / 2
    right = left + 224
    lower = upper + 224
    ins_crop = ins.crop((left, upper, right, lower))
    
    np_image = np.array(ins_crop) / 255
    np_image -= np.array ([0.485, 0.456, 0.406]) 
    np_image /= np.array ([0.229, 0.224, 0.225])
    
    np_image= np_image.transpose ((2,0,1))
    return np_image

def predict(image_path, model, gpu, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    device = torch.device("cuda" if gpu == True else "cpu")
    model.to(device)
    
    np_image = process_image(image_path)
    if (gpu == True):
        tensor = torch.from_numpy (np_image).type(torch.cuda.FloatTensor)
    else:
        tensor = torch.from_numpy (np_image).type(torch.FloatTensor)
    inputs = tensor.unsqueeze(dim = 0)
    with torch.no_grad():
        logps = model(inputs)
    ps = torch.exp(logps)
    probs, classes = ps.topk(topk)
    probs = probs.cpu()
    classes = classes.cpu()
    
    probs = probs.numpy().tolist()[0]
    classes = classes.numpy().tolist()[0]
        
    mapping = {val: key for key, val in    
                    model.class_to_idx.items()}
    labels = [mapping [i] for i in classes]
    return probs, labels