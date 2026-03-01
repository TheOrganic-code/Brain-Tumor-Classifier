import torch
from torch import nn, optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#!pip install opendatasets :) if you are using a notebook (I initially coded this through google colab!)
import opendatasets as od
import pandas
od.download('https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes') #aha google colab yet again
BASE_DATA_PATH ='./brain-tumor-mri-images-17-classes/'

tf=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# 1. Loadinggg
full_dataset=datasets.ImageFolder(BASE_DATA_PATH, tf)
total_size=len(full_dataset)

# 2. Defining the sizes for split :p
train_split_ratio=0.80
train_size=int(train_split_ratio * total_size)
test_size=total_size - train_size # The remaining images go to the test set

# 3. Random split so that we create two overlapping subsets!
train_set, test_set=random_split(full_dataset, [train_size, test_size])

# 4. Create separate DataLoaders for each subset
train_dl=DataLoader(
    train_set,  
    batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)
test_dl=DataLoader(
    test_set,  
    batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)
# the CNN (very chonky indeed)
model=nn.Sequential(
    nn.Conv2d(3,32,3,1,1),nn.ReLU(),nn.MaxPool2d(2),
    nn.Conv2d(32,64,3,1,1),nn.ReLU(),nn.MaxPool2d(2),
    nn.Conv2d(64,128,3,1,1),nn.ReLU(),nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128*16*16,256),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(256,17)
).to(device)
# ofcourse we are gonna use the GOAT optimizer here
opt=optim.AdamW(model.parameters(), 1e-4)
loss_fn=nn.CrossEntropyLoss()
#training set up
model.train()

for epoch in range(20):
  running_loss=0
  for x,y in train_dl:
    opt.zero_grad()

    loss=loss_fn(model(x.to(device)),y.to(device))
    loss.backward()
    running_loss+=loss
    opt.step()
  print(f'epoch {epoch+1}: Loss was {running_loss}')
#error and validation calculations
model.eval()
test_loss,correct=0.0,0
with torch.no_grad():
  for x,y in test_dl:
    x,y=x.to(device),y.to(device)
    logits=model(x)
    test_loss+=loss_fn(logits,y).item()*y.size(0)

    preds=logits.argmax(dim=1)
    correct+=(preds==y).sum().item()
test_loss/=len(test_dl.dataset)
accuracy=100*correct/len(test_dl.dataset)

print('Test loss: ',test_loss,' Test accuracy: ',accuracy,'%')

