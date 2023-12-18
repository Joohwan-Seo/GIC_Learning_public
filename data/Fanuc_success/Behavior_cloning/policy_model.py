import torch
import torch.nn as nn
import torch.nn.functional as F

class BCPolicy(nn.Module):
   def __init__(self, input_shape=6, output_shape=6):
      super().__init__()
      self.fc0 = nn.Linear(input_shape,128)
      self.fc1 = nn.Linear(128,128)
      self.fc2 = nn.Linear(128,128)
      self.last_fc = nn.Linear(128,output_shape)

      self.relu1 = nn.ReLU()
      self.relu2 = nn.ReLU()
      self.relu3 = nn.ReLU()
      self.tanh4 = nn.Tanh()

   def forward(self, obs):
      x = self.fc0(obs)
      x = self.relu1(x)
      x = self.fc1(x)
      x = self.relu2(x)
      x = self.fc2(x)
      x = self.relu3(x)
      x = self.last_fc(x)
      out = self.tanh4(x)
      return out
   


class BCPolicyLarger(nn.Module):
   def __init__(self, input_shape=6, output_shape=6):
      super().__init__()
      self.fc0 = nn.Linear(input_shape,128)
      self.fc1 = nn.Linear(128,128)
      self.fc2 = nn.Linear(128,128)
      self.fc3 = nn.Linear(128,128)
      self.fc4 = nn.Linear(128,128)
      self.fc5 = nn.Linear(128,128)
      self.fc6 = nn.Linear(128,128)
      self.last_fc = nn.Linear(128,output_shape)

      self.relu0 = nn.ReLU()
      self.relu1 = nn.ReLU()
      self.relu2 = nn.ReLU()
      self.relu3 = nn.ReLU()
      self.relu4 = nn.ReLU()
      self.relu5 = nn.ReLU()
      self.relu6 = nn.ReLU()
      self.tanh4 = nn.Tanh()

   def forward(self, obs):
      x = self.fc0(obs)
      x = self.relu0(x)

      x = self.fc1(x)
      x = self.relu1(x)

      x = self.fc2(x)
      x = self.relu2(x)

      x = self.fc3(x)
      x = self.relu3(x)

      x = self.fc4(x)
      x = self.relu4(x)

      x = self.fc5(x)
      x = self.relu5(x)

      x = self.fc6(x)
      x = self.relu6(x)

      x = self.last_fc(x)
      out = self.tanh4(x)
      return out