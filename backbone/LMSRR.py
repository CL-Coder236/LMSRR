import torch
from torch import nn
from torch.nn import functional as F
from transformers import ViTModel
from backbone import MammothBackbone, register_backbone
import torchvision.transforms as transforms

EPS_BATCH_NORM = 1e-4

class GumbelLayerSelector(nn.Module):
    def __init__(self, num_layers=5, tau=0.1):
        super().__init__()
        self.latent_dim = 2
        self.categorical_dim = 1  
        self.num_layers = num_layers
        self.tau = tau
        
        self.selector_weights = nn.ParameterList([
            nn.Parameter(torch.randn(2)) for _ in range(num_layers)
        ])

    def sample_gumbel(self,shape, device, eps=1e-20):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)


    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.device)
        return F.softmax(y / temperature, dim=-1)


    def gumbel_softmax(self, logits, temperature, hard=False):
        y = self.gumbel_softmax_sample(logits, temperature)
        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(*shape)

    def forward(self, layer_idx):

        logits = self.selector_weights[layer_idx]
        probs = self.gumbel_softmax(logits, self.tau)

        return probs[0]

class MultiScaleFusion(nn.Module):
    def __init__(self,num_feature: int):
        super().__init__()
        self.num_feature = num_feature
        self.conv1 = nn.Conv1d(
            in_channels=self.num_feature, 
            out_channels=1, 
            kernel_size=3, 
            padding=1 
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.num_feature, 
            out_channels=1, 
            kernel_size=5, 
            padding=2  
        )
        
        self.fusion_weight = nn.Parameter(torch.ones(3))  
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv1.bias, 0.1)
        nn.init.constant_(self.conv2.bias, 0.1)
    
    def forward(self, x):

        branch1 = self.conv1(x)
        branch2 = self.conv2(x)
        weights = F.softmax(self.fusion_weight, dim=0)  
        fused = weights[0]*branch1 + weights[1]*branch2 
        fused = fused.squeeze(1)
        return fused, weights  

class LMSRR(MammothBackbone):

    def __init__(self, num_classes: int) -> None:

        super(LMSRR, self).__init__()
        self.device = "cpu"
        self.num_classes = num_classes
        self.vitProcess = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
        vit_paths = ["google/vit-base-patch16-224","google/vit-base-patch16-224-in21k"]

        self.vit_models = nn.ModuleList([ViTModel.from_pretrained(path) for path in vit_paths])

        self.unfreezing_layer = 3
        self.features = []
        self.selectors = nn.ModuleList([
            GumbelLayerSelector(num_layers=self.unfreezing_layer)
            for _ in range(len(vit_paths))
        ])
        self.weights = []
        for vit_model in self.vit_models:
            for param in vit_model.parameters():
                param.requires_grad = False
            encoder_layers = vit_model.encoder.layer
            for layer in encoder_layers[-self.unfreezing_layer:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        self.multiScaleFusion = MultiScaleFusion(len(vit_paths))
        self.classifier = nn.Linear(768 , num_classes)  
        

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        self.features = [] 

        if(x.shape[-2:] != (224, 224)):
            x = self.vitProcess(x)
        if x.shape[1] == 1:  
            x = x.repeat(1, 3, 1, 1)  
            
        vit_features = []
        for vit_model in self.vit_models:
            outputs = vit_model(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states  
            self.features.append(hidden_states[-self.unfreezing_layer:])  
            
            class_token = outputs.last_hidden_state[:, 0, :] 
            vit_features.append(class_token)
       
        vit_combined = torch.stack(vit_features, dim=1)  
        weighted_features, weights = self.multiScaleFusion(vit_combined)
        logits = self.classifier(weighted_features)

        if returnt == 'feature':
            return self.features
        if returnt == 'out':
            return logits
        raise NotImplementedError("Unknown return type. Must be in ['out', 'features'] but got {}".format(returnt))


@register_backbone('lmsrr')
def lmsrr(num_classes: int):

    return LMSRR(num_classes=num_classes)
