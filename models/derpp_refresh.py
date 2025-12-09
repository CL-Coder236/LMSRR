import torch
from torch.nn import functional as F
import torch.nn as nn
from copy import deepcopy
import copy
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer

epsilon = 1E-20

class Derpprefresh(ContinualModel):
    NAME = 'derpp_refresh'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        
        super(Derpprefresh, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.net.to(self.device)
        self.buffer = Buffer(self.args.buffer_size)

        self.temp = copy.deepcopy(self.net)
        self.temp_opt = torch.optim.SGD(self.temp.parameters(), lr=0.01)

        lr = self.args.lr
        weight_decay = 0.0001
        self.delta = 0.00001
        self.tau = 0.00001

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = {}
        for name, param in self.net.named_parameters():
            self.fish[name] = torch.zeros_like(param)

        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        
        
    def unlearn(self, inputs, labels):
        self.temp.load_state_dict(self.net.state_dict())
        self.temp.train()
        outputs = self.temp(inputs)
        loss = - F.cross_entropy(outputs, labels)
        self.temp_opt.zero_grad()
        loss.backward()
        self.temp_opt.step()

        for (model_name, model_param), (temp_name, temp_param) in zip(self.net.named_parameters(), self.temp.named_parameters()):
            if model_param.requires_grad:
                weight_update = temp_param - model_param
                model_param_norm = model_param.norm()
                weight_update_norm = weight_update.norm() + epsilon
                norm_update = model_param_norm / weight_update_norm * weight_update
                identity = torch.ones_like(self.fish[model_name])
                with torch.no_grad():
                    model_param.add_(self.delta * torch.mul(1.0/(identity + 0.001*self.fish[model_name]), norm_update + 0.001*torch.randn_like(norm_update)))
                    
    def end_task(self, dataset):
        self.temp.load_state_dict(self.net.state_dict())
        fish = {}
        for name, param in self.temp.named_parameters():
            fish[name] = torch.zeros_like(param)

        for j, data in enumerate(dataset.train_loader):
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.temp_opt.zero_grad()
                output = self.temp(ex.unsqueeze(0))

                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                for name, param in self.temp.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fish[name] +=  exp_cond_prob * param.grad ** 2

        for name, param in self.temp.named_parameters():
            if param.requires_grad and param.grad is not None:
                fish[name] /= (len(dataset.train_loader) * self.args.batch_size)
       
        for key in self.fish:
                self.fish[key] *= self.tau
                self.fish[key] += fish[key]

        self.checkpoint = self.net.get_params().data.clone()
        self.temp_opt.zero_grad()
                           
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.unlearn(inputs=inputs, labels=labels)
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)


        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform,device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform,device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item()
    