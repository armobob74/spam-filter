import torch.nn
class BasicNet(torch.nn.Module):
    '''Classify  vectors as spam or ham'''
    def __init__(self):
        super(BasicNet,self).__init__()
        self.dense1 = torch.nn.Linear(46,32)
        self.dense2 = torch.nn.Linear(32,32)
        self.dense3 = torch.nn.Linear(32,1)
        self.act1 = torch.nn.Sigmoid()
        self.act2 = torch.nn.Sigmoid()
        self.act3 = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        x = self.dense3(x)
        x = self.act3(x)
        return x
