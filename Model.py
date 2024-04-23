import torch

# the shape if KGV_NN
structure_size = [1, 28, 28]


class Encoder(torch.nn.Module):
    """
    A basic model for MNIST.
    """

    def __init__(self, task_output_size=10, ssize=structure_size):
        """
        :param task_output_size: for the classification task of MNIST.
        :param ssize: structure size.
        """
        super(Encoder, self).__init__()
        # encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )
        # generator (generate KGV_NN)
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, ssize[0] * ssize[1] * ssize[2]),
            torch.nn.Sigmoid()  # activated with Sigmoid to make sure the value is in [0,1]
        )
        # classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, task_output_size),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        structure = self.generator(embedding)
        classification = self.classifier(embedding)
        return classification, structure


class Decoder(torch.nn.Module):

    def __init__(self, ssize=structure_size):
        super(Decoder, self).__init__()
        # decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(ssize[0] * ssize[1] * ssize[2], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x_p = self.decoder(x)
        return x_p
