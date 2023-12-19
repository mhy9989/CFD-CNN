from torch import nn

def relu():
    return nn.ReLU(),


def gelu():
    return nn.GELU(),


def tanh():
    return nn.Tanh(),


def leakyrelu():
    return nn.LeakyReLU(),


def sigmoid():
    return nn.Sigmoid(),

