import torch
import torchvision
from typing import List
from MNISTNet import MNISTNet
from torchvision import transforms
from PIL import Image


tran = transforms.Compose([
    transforms.ToTensor()
])


model = torch.load("/Users/aidarazizov/PycharmProjects/teest/mnist_full_model.pt")
model.eval()

img = Image.open("/MNIST_formatted.png")
img = tran(img)
img *= 255.

out = model(img.unsqueeze(0))
print('MODEl')
print('\t', out)
print('\t', out.argmax().item())



sample = torch.rand((1, 1, 28, 28))
trace_model = torch.jit.trace(model, img.unsqueeze(0))

trace_model.save('trace_model.pt')
script_model = torch.jit.script(model)

out2 = script_model.forward(img.unsqueeze(0))
print('scripted-MODEl')
print('\t', out2)
print('\t', out2.argmax().item())

