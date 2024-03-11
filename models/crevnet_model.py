import torch
from torch import nn
from torch.autograd import Variable

from modules import zig_revpredictor, autoencoder


class CrevNet_Model(nn.Module):
    r"""CrevNet Model

    Implementation of `Efficient and Information-Preserving Future Frame Prediction
    and Beyond <https://openreview.net/forum?id=B1eY_pVYvB>`_.

    Notice: CrevNet Model requires `batch_size` == `val_batch_size`, or it will raise
    """

    def __init__(self, in_shape, rnn_size, batch_size, predictor_rnn_layers,
                 data_previous, data_after, n_eval, **kwargs):
        super(CrevNet_Model, self).__init__()
        T, channels, image_height, image_width = in_shape
        self.rnn_size = rnn_size
        self.n_eval = n_eval
        self.data_previous = data_previous
        self.data_after = data_after

        self.framepredictor = zig_revpredictor(
            rnn_size, rnn_size, rnn_size, predictor_rnn_layers, batch_size)

        self.encoder = autoencoder(nBlocks=[4,5,3], nStrides=[1, 2, 2],
                            nChannels=None, init_ds=2,
                            dropout_rate=0., affineBN=True,
                            in_shape=[channels, image_height, image_width],
                            mult=2)
        self.criterion = nn.MSELoss()

    def forward(self, x, training=True, **kwargs):
        B, T, C, H, W = x.shape

        input = []
        for j in range(self.n_eval):
            k1 = x[:, j].unsqueeze(2)
            k2 = x[:, j + 1].unsqueeze(2)
            k3 = x[:, j + 2].unsqueeze(2)
            input.append(torch.cat((k1,k2,k3), 2))

        loss = 0
        self.framepredictor.hidden = self.framepredictor.init_hidden()
        memo = Variable(torch.zeros(B, self.rnn_size, 3, H // 8, W // 8).cuda())
        for i in range(1, self.data_previous + self.data_after):
            h = self.encoder(input[i - 1], True)
            try:
                h_pred, memo = self.framepredictor((h, memo))
            except RuntimeError:
                assert False and "CrevNet Model requires `batch_size` == `val_batch_size`"
            x_pred = self.encoder(h_pred, False)
            if kwargs.get('return_loss', True):
                loss += (self.criterion(x_pred, input[i]))

        if training is True:
            return loss
        else:
            gen_seq = []
            self.framepredictor.hidden = self.framepredictor.init_hidden()
            memo = torch.zeros(B, self.rnn_size, 3, H // 8, W // 8).cuda()
            x_in = input[self.data_previous-1]
            for i in range(self.data_previous, self.n_eval):
                h = self.encoder(x_in)
                h_pred, memo = self.framepredictor((h, memo))
                if i == self.data_previous:
                    x_in = self.encoder(h_pred, False).detach()
                    x_in[:, :, 0] = input[i][:, :, 0]
                    x_in[:, :, 1] = input[i][:, :, 1]
                elif i == self.data_previous + 1:
                    x_in = self.encoder(h_pred, False).detach()
                    x_in[:, :, 0] = input[i][:, :, 0]
                else:
                    x_in = self.encoder(h_pred, False).detach()
                gen_seq.append(x_in[:, 0, 2][:, None, ...])

            return torch.stack(gen_seq, dim=1), loss
