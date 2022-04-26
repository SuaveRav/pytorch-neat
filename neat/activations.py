import logging

import torch


def sigmoid(x):
    return torch.sigmoid(x)


def tanh(x):
    return torch.tanh(x)

def relu(x):
    return torch.relu(x)

def relu(x):
    return torch.relu(x)


def tanhshrink(x):
    return torch.nn.Tanhshrink()(x)


def logsigmoid(x):
    return torch.nn.LogSigmoid()(x)


def negate(x):
    return -x


class Activations:

    def __init__(self):
        self.functions = dict(
            sigmoid=sigmoid,
            tanh=tanh,
            relu=relu,
            tanhshrink=tanhshrink,
            logsigmoid=logsigmoid,
            negate=negate,
        )

    def get(self, func_name):
        return self.functions.get(func_name, None)
