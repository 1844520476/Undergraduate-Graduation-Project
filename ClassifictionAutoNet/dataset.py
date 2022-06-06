# dataset.py
from torchvision import datasets

training_set = datasets.ImageFolder(root=r'C:\Users\cleste\Desktop\孪生网络（少样本分类）\datasets\mac-set\train-set')
testing_set = datasets.ImageFolder(root=r'C:\Users\cleste\Desktop\孪生网络（少样本分类）\datasets\mac-set\test-set')
for i in range(len(training_set)):
    print(i, training_set[i])
for i in range(len(testing_set)):
    print(i, testing_set[i])
