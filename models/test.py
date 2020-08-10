import torch 
import torch.nn as nn

loss = nn.MSELoss(reduction='none')
input = torch.ones(3, 5, requires_grad=True)
target = torch.zeros(3, 5)
output = loss(input, target)
output.backward()
print("output:", output)