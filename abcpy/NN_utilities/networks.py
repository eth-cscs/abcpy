from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.distributions import normal, laplace, cauchy
from torchtyping import TensorType, patch_typeguard

patch_typeguard()

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


class ScalerAndNet(nn.Module):
    """Defines a nn.Module class that wraps a scaler and a neural network, and applies the scaler before passing the
    data through the neural network."""

    def __init__(self, net, scaler):
        """"""
        super().__init__()
        self.net = net
        self.scaler = scaler

    def forward(self, x):
        """"""
        x = torch.tensor(self.scaler.transform(x), dtype=torch.float32).to(next(self.net.parameters()).device)
        return self.net(x)


class DiscardLastOutputNet(nn.Module):
    """Defines a nn.Module class that wraps a scaler and a neural network, and applies the scaler before passing the
    data through the neural network. Next, the """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net(x)
        if len(x.shape) == 1:
            return x[0:-1]
        if len(x.shape) == 2:
            return x[:, 0:-1]
        if len(x.shape) == 3:
            return x[:, :, 0:-1]


def createDefaultNN(input_size, output_size, hidden_sizes=None, nonlinearity=None, batch_norm_last_layer=False,
                    batch_norm_last_layer_momentum=0.1):
    """Function returning a fully connected neural network class with a given input and output size, and optionally
    given hidden layer sizes (if these are not given, they are determined from the input and output size in a heuristic
    way, see below).

    In order to instantiate the network, you need to write:

        >>> createDefaultNN(input_size, output_size)()

    as the function returns a class, and () is needed to instantiate an object.

    If hidden_sizes is None, three hidden layers are used with the following sizes:
    ``[int(input_size * 1.5), int(input_size * 0.75 + output_size * 3), int(output_size * 5)]``

    Note that the nonlinearity here is as an object or a functional, not a class, eg:
        nonlinearity =  nn.Softplus()
    or:
        nonlinearity =  nn.functional.softplus

    """

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

            # define the batch_norm:
            if batch_norm_last_layer:
                self.bn_out = nn.BatchNorm1d(output_size, affine=False, momentum=batch_norm_last_layer_momentum)

        def forward(self, x):

            if nonlinearity is None:
                nonlinearity_fcn = F.relu
            else:
                nonlinearity_fcn = nonlinearity

            if not hasattr(self,
                           "fc_hidden"):  # it means that hidden sizes was provided and the length of the list was 0
                return self.fc_in(x)

            x = nonlinearity_fcn(self.fc_in(x))
            for i in range(len(self.fc_hidden)):
                x = nonlinearity_fcn(self.fc_hidden[i](x))

            x = self.fc_out(x)

            if batch_norm_last_layer:
                x = self.bn_out(x)

            return x

    return DefaultNN


def createDefaultNNWithDerivatives(input_size, output_size, hidden_sizes=None, nonlinearity=None,
                                   first_derivative_only=False):
    """Function returning a fully connected neural network class with a given input and output size, and optionally
    given hidden layer sizes (if these are not given, they are determined from the input and output size with some
    expression. This neural network is capable of computing the first and second derivatives of output with respect to
    input along with the forward pass.

    All layers in this neural network are linear.

        >>> createDefaultNN(input_size, output_size)()

    as the function returns a class, and () is needed to instantiate an object.

    If hidden_sizes is None, three hidden layers are used with the following sizes:
    ``[int(input_size * 1.5), int(input_size * 0.75 + output_size * 3), int(output_size * 5)]``

    Note that the nonlinearity here is passed as a class, not an object, eg:
        nonlinearity =  nn.Softplus
    """

    if nonlinearity in [torch.nn.Softsign, torch.nn.Tanhshrink]:
        raise RuntimeError("The implementation of forward derivatives does not work with Tanhshrink and "
                           "Softsign nonlinearities.")

    class DefaultNNWithDerivatives(nn.Module):
        """Neural network class with sizes determined by the upper level variables."""

        def __init__(self):
            super(DefaultNNWithDerivatives, self).__init__()
            # put some fully connected layers:

            if nonlinearity is None:  # default nonlinearity
                non_linearity = nn.ReLU
            else:
                non_linearity = nonlinearity  # need to change name otherwise it gives Error

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
                self.nonlinearity_in = non_linearity()

                # define now the hidden layers
                self.fc_hidden = nn.ModuleList()
                self.nonlinearities_hidden = nn.ModuleList()
                for i in range(len(hidden_sizes_list) - 1):
                    self.fc_hidden.append(nn.Linear(hidden_sizes_list[i], hidden_sizes_list[i + 1]))
                    self.nonlinearities_hidden.append(non_linearity())
                self.fc_out = nn.Linear(hidden_sizes_list[-1], output_size)

        def forward(self, x):

            if not hasattr(self,
                           "fc_hidden"):  # it means that hidden sizes was provided and the length of the list was 0, ie the
                return self.fc_in(x)

            x = self.fc_in(x)
            x1 = self.nonlinearity_in(x)

            for i in range(len(self.fc_hidden)):
                x = self.fc_hidden[i](x1)
                x1 = self.nonlinearities_hidden[i](x)

            x = self.fc_out(x1)

            return x

        def forward_and_derivatives(self, x):

            # initialize the derivatives:
            f = self.fc_in.weight.unsqueeze(0).repeat(x.shape[0], 1, 1).transpose(2, 1).transpose(0,
                                                                                                  1)  # one for each element of the batch
            if not first_derivative_only:
                s = torch.zeros_like(f)

            if not hasattr(self, "fc_hidden"):
                # it means that hidden sizes was provided and the length of the list was 0, ie the net is a single layer.
                if first_derivative_only:
                    return self.fc_in(x), f.transpose(0, 1)
                else:
                    return self.fc_in(x), f.transpose(0, 1), s.transpose(0, 1)

            x = self.fc_in(x)
            x1 = self.nonlinearity_in(x)

            for i in range(len(self.fc_hidden)):
                z = x1.grad_fn(torch.ones_like(x1))  # here we repeat some computation from the above line
                # z = grad(x1, x, torch.ones_like(x1), create_graph=True)[0]  # here we repeat some computation from the above line
                # you need to update first the second derivative, as you need the first derivative at previous layer
                if not first_derivative_only:
                    s = z * s + grad(z, x, torch.ones_like(z), retain_graph=True)[0] * f ** 2
                f = z * f
                f = F.linear(f, self.fc_hidden[i].weight)
                if not first_derivative_only:
                    s = F.linear(s, self.fc_hidden[i].weight)

                x = self.fc_hidden[i](x1)
                x1 = self.nonlinearities_hidden[i](x)

            z = x1.grad_fn(torch.ones_like(x1))  # here we repeat some computation from the above line
            # z = grad(x1, x, torch.ones_like(x1), create_graph=True)[0]  # here we repeat some computation from the above line
            # you need to update first the second derivative, as you need the first derivative at previous layer
            if not first_derivative_only:
                s = z * s + grad(z, x, torch.ones_like(z), retain_graph=True)[0] * f ** 2
            f = z * f
            f = F.linear(f, self.fc_out.weight)
            if not first_derivative_only:
                s = F.linear(s, self.fc_out.weight)

            x = self.fc_out(x1)

            if first_derivative_only:
                return x, f.transpose(0, 1)
            else:
                return x, f.transpose(0, 1), s.transpose(0, 1)

        def forward_and_full_derivatives(self, x):
            """This computes jacobian and full Hessian matrix"""

            # initialize the derivatives (one for each element of the batch)
            f = self.fc_in.weight.unsqueeze(0).repeat(x.shape[0], 1, 1).transpose(2, 1).transpose(0, 1)
            H = torch.zeros((f.shape[0], *f.shape)).to(f)  # hessian has an additional dimension wrt f

            if not hasattr(self, "fc_hidden"):
                # it means that hidden sizes was provided and the length of the list was 0, ie the net is a single layer
                return self.fc_in(x), f.transpose(0, 1), H.transpose(0, 2)

            x = self.fc_in(x)
            x1 = self.nonlinearity_in(x)

            for i in range(len(self.fc_hidden)):
                z = x1.grad_fn(torch.ones_like(x1))  # here we repeat some computation from the above line
                # print("H", H.shape, "z", z.shape, "z'", grad(z, x, torch.ones_like(z), retain_graph=True)[0].shape, "f", f.shape)
                # z = grad(x1, x, torch.ones_like(x1), create_graph=True)[0]  # here we repeat some computation from the above line
                # you need to update first the second derivative, as you need the first derivative at previous layer
                H = z * H + grad(z, x, torch.ones_like(z), retain_graph=True)[0] * torch.einsum('ibo,jbo->ijbo', f, f)
                f = z * f
                f = F.linear(f, self.fc_hidden[i].weight)
                H = F.linear(H, self.fc_hidden[i].weight)

                x = self.fc_hidden[i](x1)
                x1 = self.nonlinearities_hidden[i](x)

            z = x1.grad_fn(torch.ones_like(x1))  # here we repeat some computation from the above line
            # z = grad(x1, x, torch.ones_like(x1), create_graph=True)[0]  # here we repeat some computation from the above line
            # you need to update first the second derivative, as you need the first derivative at previous layer
            H = z * H + grad(z, x, torch.ones_like(z), retain_graph=True)[0] * torch.einsum('ibo,jbo->ijbo', f, f)
            f = z * f
            f = F.linear(f, self.fc_out.weight)
            H = F.linear(H, self.fc_out.weight)
            x = self.fc_out(x1)

            return x, f.transpose(0, 1), H.transpose(0, 2)

    return DefaultNNWithDerivatives


def createGenerativeFCNN(input_size, output_size, hidden_sizes=None, nonlinearity=None):
    """Function returning a fully connected neural network class with a given input and output size, and optionally
    given hidden layer sizes (if these are not given, they are determined from the input and output size with some
    expression. With respect to the one above, the NN here has two inputs,
    which are the context x and the auxiliary variable z in the case of this being used for a generative model.

    In order to instantiate the network, you need to write: createGenerativeFCNN(input_size, output_size)() as the function
    returns a class, and () is needed to instantiate an object.

    `input_size` needs to match n_parameters in the model plus the size of auxiliary variable z.

    Note that the nonlinearity here is as an object or a functional, not a class, eg:
        nonlinearity =  nn.Softplus()
    or:
        nonlinearity =  nn.functional.softplus
    """

    class GenerativeFCNN(nn.Module):
        """Neural network class with sizes determined by the upper level variables."""

        def __init__(self):
            super(GenerativeFCNN, self).__init__()
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

            self.nonlinearity_fcn = F.relu if nonlinearity is None else nonlinearity

        def forward(self, parameters: TensorType["batch_size", "n_parameters"],
                    noise: TensorType["batch_size", "number_generations", "size_auxiliary_variable"]) \
                -> TensorType["batch_size", "number_generations", "output_size"]:
            # this network just flattens the context and concatenates it to the auxiliary variable, for each possible
            # auxiliary variable for each batch element, and then uses a FCNN.
            # input size of the FCNN must be equal to window_size * data_size + size_auxiliary_variable

            batch_size, n_parameters = parameters.shape
            batch_size, number_generations, size_auxiliary_variable = noise.shape

            # can add argument `output_size=batch_size * number_generations` with torch 1.10
            repeated_context = torch.repeat_interleave(parameters, repeats=number_generations, dim=0).reshape(
                batch_size, number_generations, n_parameters)

            input_tensor = torch.cat((repeated_context, noise), dim=-1)

            if not hasattr(self,
                           "fc_hidden"):  # it means that hidden sizes was provided and the length of the list was 0
                input_tensor = self.fc_in(input_tensor)
                return input_tensor

            input_tensor = self.nonlinearity_fcn(self.fc_in(input_tensor))
            for i in range(len(self.fc_hidden)):
                input_tensor = self.nonlinearity_fcn(self.fc_hidden[i](input_tensor))

            return self.fc_out(input_tensor)

    return GenerativeFCNN


class ConditionalGenerativeNet(nn.Module):
    """This is a class wrapping a net which takes as an input a conditioning variable and an auxiliary variable,
    concatenates then and then feeds them through the NN. The auxiliary variable generation is done whenever the
    forward method is called."""

    def __init__(self, net, size_auxiliary_variable: Union[int, torch.Size],
                 number_simulations_per_forward_call: int = 1, seed: int = 42, base_measure: str = "normal"):
        super(ConditionalGenerativeNet, self).__init__()
        self.net = net  # net has to be able to take input size `size_auxiliary_variable + dim(x)`
        if isinstance(size_auxiliary_variable, int):
            size_auxiliary_variable = torch.Size([size_auxiliary_variable])
        self.size_auxiliary_variable = size_auxiliary_variable
        self.number_simulations_per_forward_call = number_simulations_per_forward_call
        # set the seed of the random number generator for ensuring reproducibility:
        torch.random.manual_seed(seed)
        distribution_dict = {"normal": normal.Normal, "laplace": laplace.Laplace, "cauchy": cauchy.Cauchy}
        if base_measure not in distribution_dict.keys():
            raise NotImplementedError("Base measure not available")
        self.distribution = distribution_dict[base_measure](loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))

    def forward(self, context: TensorType["batch_size", "n_parameters"], n_simulations: int = None) -> \
    TensorType["batch_size", "number_generations", "data_size"]:
        """This returns a stacked torch tensor for outputs. For now, I use different random auxiliary variables for
            each element in the batch size, but in principle you could take the same."""

        # you basically need to generate self.number_simulations_per_forward_call forward simulations with a different
        # auxiliary variable z, all with the same conditioning variable x, for each conditioning variable x:
        if n_simulations is None:
            n_simulations = self.number_simulations_per_forward_call

        if context.ndim == 2:
            batch_size, n_parameters = context.shape
        else:
            raise NotImplementedError

        # generate the auxiliary variables; use different noise for each batch element
        z = self.distribution.sample(torch.Size([batch_size, n_simulations]) + self.size_auxiliary_variable).to(
            device="cuda" if next(self.parameters()).is_cuda else "cpu").squeeze(-1)

        return self.net(context, z)
