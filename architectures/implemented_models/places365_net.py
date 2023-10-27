from functools import reduce
import os  
import requests

import torch
import torch.nn as nn

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))
    

class ResNet50Places365(nn.Module):
    def __init__(self):
        super(ResNet50Places365, self).__init__()
        self.model =  nn.Sequential( # Sequential,
            nn.Conv2d(3,64,(7, 7),(2, 2),(3, 3),1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3),(2, 2),(1, 1)),
        nn.Sequential( # Sequential,
            nn.Sequential( # Sequential,
                LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                nn.Conv2d(64,64,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
            ),
            nn.Sequential( # Sequential,
                nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
            ),
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,64,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                ),
                Lambda(lambda x: x), # Identity,
                ),
                LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,64,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                ),
            Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
        ),
        nn.Sequential( # Sequential,
            nn.Sequential( # Sequential,
                nn.BatchNorm2d(256),
                nn.ReLU(),
                LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128,128,(3, 3),(2, 2),(1, 1),1,1,bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                ),
            nn.Sequential( # Sequential,
                nn.Conv2d(256,512,(1, 1),(2, 2),(0, 0),1,1,bias=False),
            ),
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(512,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                ),
            Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
            ),
            Lambda(lambda x: x), # Identity,
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
            ),
            Lambda(lambda x: x), # Identity,
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
        ),
        ),
        nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            nn.BatchNorm2d(512),
            nn.ReLU(),
            LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
                nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,256,(3, 3),(2, 2),(1, 1),1,1,bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
            ),
            nn.Sequential( # Sequential,
                nn.Conv2d(512,1024,(1, 1),(2, 2),(0, 0),1,1,bias=False),
            ),
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
        ),
        nn.Sequential( # Sequential,
        LambdaMap(lambda x: x, # ConcatTable,
        nn.Sequential( # Sequential,
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
            ),
            Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
            ),
            Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                ),
                Lambda(lambda x: x), # Identity,
                ),
                LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
            nn.Sequential( # Sequential,
                LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(1024,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                ),
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
        ),
        Lambda(lambda x: x), # Identity,
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
        ),
        ),
        nn.Sequential( # Sequential,
            nn.Sequential( # Sequential,
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512,512,(3, 3),(2, 2),(1, 1),1,1,bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
            ),
            nn.Sequential( # Sequential,
                nn.Conv2d(1024,2048,(1, 1),(2, 2),(0, 0),1,1,bias=False),
            ),
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
        ),
        Lambda(lambda x: x), # Identity,
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
        ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.Conv2d(2048,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
            ),
            Lambda(lambda x: x), # Identity,
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
        ),
        ),
        nn.BatchNorm2d(2048),
        nn.ReLU(),
        nn.AvgPool2d((7, 7),(1, 1)),
        Lambda(lambda x: x.view(x.size(0),-1)), # View,
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2048,365)), # Linear,
        )
    
    def forward(self,x):
        return self.model(x)
        

def load_state_dict_compatibly(model, state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = "model." + key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)

    return model


def get_resnet50_places365(pretrained=True):
    model = ResNet50Places365()

    if not os.path.exists(r"checkpoints"):
        os.makedirs(r"checkpoints")

    if pretrained:

        # Check if the file exists in the current directory
        if not os.path.exists(r"checkpoints\resnet50_places365.pth"):
            # Download the file using requests
            file_url = "https://drive.google.com/uc?id=1qO96TKv2zeLI8Poi2ISKVCcLrrUOMRaz"
            r = requests.get(file_url)
            print("~ Downloading weights")
            with open(r"checkpoints\resnet50_places365.pth", "wb") as f:
                f.write(r.content)
        
        # Load the weights into the model
        state_dict = torch.load(r"checkpoints\resnet50_places365.pth")
        model = load_state_dict_compatibly(model, state_dict)

        
    return model
