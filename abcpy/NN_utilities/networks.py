import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    """ This is used in the contrastive distance learning. It is a network wrapping a standard neural network and
    feeding two samples through it at once.

    From https://github.com/adambielski/siamese-triplet"""

    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    """ This is used in the triplet distance learning. It is a network wrapping a standard neural network and
    feeding three samples through it at once.

    From https://github.com/adambielski/siamese-triplet"""

    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


def createDefaultNN(input_size, output_size, hidden_sizes=None, nonlinearity=None):
    """Function returning a fully connected neural network class with a given input and output size, and optionally
    given hidden layer sizes (if these are not given, they are determined from the input and output size with some
    expression.

    In order to instantiate the network, you need to write: createDefaultNN(input_size, output_size)() as the function
    returns a class, and () is needed to instantiate an object."""

    class DefaultNN(nn.Module):
        """Neural network class with sizes determined by the upper level variables."""

        def __init__(self):
            super(DefaultNN, self).__init__()
            # put some fully connected layers:

            if hidden_sizes is not None and len(hidden_sizes) == 0:
                # it is effectively a linear network
                self.fc_in = nn.Linear(input_size, output_size)

            else:
                if hidden_sizes is None:
                    # then set some default values for the hidden layers sizes; is this parametrization reasonable?
                    hidden_sizes_list = [int(input_size * 1.5), int(input_size * 0.75 + output_size * 3),
                                         int(output_size * 5)]

                else:
                    hidden_sizes_list = hidden_sizes

                self.fc_in = nn.Linear(input_size, hidden_sizes_list[0])

                # define now the hidden layers
                self.fc_hidden = nn.ModuleList()
                for i in range(len(hidden_sizes_list) - 1):
                    self.fc_hidden.append(nn.Linear(hidden_sizes_list[i], hidden_sizes_list[i + 1]))
                self.fc_out = nn.Linear(hidden_sizes_list[-1], output_size)

        def forward(self, x):
            if not hasattr(self,
                           "fc_hidden"):  # it means that hidden sizes was provided and the length of the list was 0
                return self.fc_in(x)

            if nonlinearity is None:
                x = F.relu(self.fc_in(x))
                for i in range(len(self.fc_hidden)):
                    x = F.relu(self.fc_hidden[i](x))
            else:
                x = nonlinearity(self.fc_in(x))
                for i in range(len(self.fc_hidden)):
                    x = nonlinearity(self.fc_hidden[i](x))

            x = self.fc_out(x)

            return x

    return DefaultNN

