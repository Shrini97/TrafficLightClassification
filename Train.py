from torchvision import transforms
from torch.utils import data
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from DataLoader import *
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

TrainLoader = TrafficLight(RootDirectory="./data/train/")
Params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 8}
TrainingGenerator = data.DataLoader(TrainLoader, **Params)

TestLoader = TrafficLight(RootDirectory="./data/test/")
Params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 8}
TestingGenerator = data.DataLoader(TestLoader, **Params)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
Model = MultiLabelClassifier(FeatureExtractor = 'resnet50').to(device)
Loss = nn.BCELoss(reduction='mean')
Optimizer = torch.optim.Adam([  {'params': Model.conv1.parameters(), 'lr': 1e-4},
                                {'params': Model.linear.parameters(), 'lr': 1e-4}
                            ], lr=1e-4, betas=[0.9, 0.999], weight_decay = 0.001)
train_losses = []
test_losses = []
epochs = 40
f1 = []
states = ["red", "yellow", "green", "green-left", "green-right", "green-solid"]

for epoch in range(1, epochs):
    predictions = []
    all_test_labels = []

    # Training
    l0 = 0
    steps = 0
    min_loss = 10
    t3 = time.time()
    
    for local_batch, local_labels in TrainingGenerator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        t1 = time.time()
        Model.zero_grad()
        
        Output = Model(local_batch)
        t2 = time.time()
        
        l = Loss(Output, local_labels)
        l.backward()
        Optimizer.step()
        
        t3 = time.time()
        print("Epoch in training :", epoch, "steps:", steps, "Forward Pass:", "%.3f" % (t2-t1), "Optimization:", "%.3f" % (t3-t2), "Loss:", l.item())
        l0+=l.item()
        steps+=1
    
    l0=l0/steps
    train_losses.append(l0)

    l0 = 0
    steps = 0
    
    
    t3 = time.time()
    for local_batch, local_labels in TestingGenerator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        t1 = time.time()
        Model.zero_grad()
        
        Output = Model(local_batch)
        t2 = time.time()
        
        l = Loss(Output, local_labels)
        t3 = time.time()
        
        
        print("Epoch in testing:", epoch, "steps:", steps, "Forward Pass:", "%.3f" % (t2-t1), "Optimization:", "%.3f" % (t3-t2), "Loss:", l.item())
        l0+=l.item()
        steps+=1

        for op, lab in zip(Output.detach().tolist(), local_labels.detach().tolist()):
            predictions.append(op)
            all_test_labels.append(lab)

    
    l0=l0/steps
    test_losses.append(l0)
    
    if l0<min_loss:
        torch.save(Model.state_dict(), "model_prunned_lr.pt")
        min_loss = l0
    
    predictions = np.array(predictions, dtype = np.float32) 
    predictions = predictions > 0.5 
    predictions = predictions.astype('float').tolist()
    
    f1.append([f1_score(all_test_labels[:][i], predictions[:][i], average='weighted') for i in range(6)])
    

plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(test_losses, label='Testing Loss', color='red')
plt.xlabel("epochs")
plt.ylabel("Average BCE loss")
plt.legend(loc="upper right")
plt.savefig("losses_prunned_lr.png")
plt.close()

for i in range(6):
    plt.plot(f1[:][i], label=states[i])

plt.xlabel("epochs")
plt.ylabel("F1 score")
plt.legend(loc="upper left")    
plt.savefig("f1_prunned_lr.png")
plt.close()