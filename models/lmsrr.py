from torch.nn import functional as F 
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import torch
import copy


#####################
class lmsrr(ContinualModel):
    NAME = 'lmsrr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='ARO weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        return parser

    
    def __init__(self, backbone, loss, args, transform,dataset=None):
        super(lmsrr,self).__init__(backbone, loss, args, transform,dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

    def begin_task(self, dataset):
        self.opt = self.get_optimizer() 
        self.old_net = copy.deepcopy(self.net)
        for param in self.old_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.net(x)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        feature_loss = torch.tensor(0.0, device=self.device) 
        outputs = self.net(inputs)
        total_loss = self.loss(outputs, labels)

        if self.current_task > 0:
            features = self.net(inputs,'feature')
            buf_features = self.old_net(inputs,'feature')
            feature_loss = 0.
            layer_count = 0
            for j,(prev_vit_feat, curr_vit_feat, selector) in enumerate(zip(buf_features, features, self.net.selectors)):
                for i, (prev_layer, curr_layer) in enumerate(zip(prev_vit_feat, curr_vit_feat)):  
                    selection_prob = selector(i)
                    feature_loss += F.mse_loss(prev_layer.detach(), curr_layer)*selection_prob
                    layer_count += selection_prob
            if layer_count > 0:
                feature_loss /= layer_count    
            total_loss += self.args.alpha * feature_loss

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            buf_loss = self.args.beta * F.mse_loss(buf_outputs, buf_logits)
            total_loss += buf_loss

        total_loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return total_loss.item()