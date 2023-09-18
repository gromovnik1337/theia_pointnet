from typing import Tuple
from typing import Any
import torch
import torch.nn.functional as F
from config import config


class RegressionNetwork(torch.nn.Module):
    """Simple MLP regression network, 2 hidden layers.
    """

    def __init__(self, in_dim: int, first_hidden: int, second_hidden: int,
                 out_dim: int):
        """
        Args:
            in_dim: Input vector dimension.
            first_hidden: Dimension of first hidden layer.
            second_hidden: Dimension of second hidden layer.
            out_dim: Output vector dimension.
        """
        super(RegressionNetwork, self).__init__()
        # Define input, hidden and output layers:
        self.input_l = torch.nn.Linear(in_dim, first_hidden)
        self.hidden_l_1 = torch.nn.Linear(first_hidden, second_hidden)
        self.hidden_l_2 = torch.nn.Linear(second_hidden, second_hidden)
        self.output_l = torch.nn.Linear(second_hidden, out_dim)

        # Define transfer (activation) function.
        self.t_function = torch.nn.Softplus()

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Defines 1 forward pass of the neural network.

        Args:
            X: Input data.

        Returns:
            Inference result in the pass.
        """
        h_1 = self.t_function(self.input_l(X))
        h_2 = self.t_function(self.hidden_l_1(h_1))
        h_3 = self.t_function(self.hidden_l_2(h_2))
        y = self.output_l(h_3)

        return y


def instantiate_regression_model(
        device: torch.device) -> Tuple[torch.nn.Module, Any, Any]:
    """Wraps the creation of an exact instance of a regression model, using loaded parameters.
    Furthemore, it also defines exact loss function and gradient optimizer, both relevant
    for models training.

    Args:
        device: PyTorch device (gpu or cpu) aka. hardware that will be used for computation.

    Returns:
        Defined model with its loss function and gradient optimizer.
    """
    # Load global config & get model parameters.
    config_file = config.EosConfig()

    in_dim = config_file.config['regression_network']['in_dim']
    first_hidden = config_file.config['regression_network']['first_hidden']
    second_hidden = config_file.config['regression_network']['second_hidden']
    out_dim = config_file.config['regression_network']['output_dim']
    lr = config_file.config['regression_network']['lr']

    # Define the model:
    model = RegressionNetwork(in_dim, first_hidden, second_hidden,
                              out_dim).to(device)
    # Define loss function and gradient optimizer.
    criterion = torch.nn.MSELoss()  # Loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer

class Tnet(torch.nn.Module):
    """T-net model, designed to predict canonical orientation
    of each point cloud, resembles larger PointNet.
    """
    def __init__(self, k=3):
      """_summary_

      Args:
          k (int, optional): _description_. Defaults to 3.
            """
        super().__init__()
        self.k=k
        self.conv1 = torch.nn.Conv1d(k,64,1)
        self.conv2 = torch.nn.Conv1d(64,128,1)
        self.conv3 = torch.nn.Conv1d(128,1024,1)
        self.fc1 = torch.nn.Linear(1024,512)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc3 = torch.nn.Linear(256,k*k)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
      

  def forward(self, input):
    # input.shape == (bs,n,3)
    bs = input.size(0)
    xb = F.relu(self.bn1(self.conv1(input)))
    xb = F.relu(self.bn2(self.conv2(xb)))
    xb = F.relu(self.bn3(self.conv3(xb)))
    pool = torch.nn.MaxPool1d(xb.size(-1))(xb)
    flat = torch.nn.Flatten(1)(pool)
    xb = F.relu(self.bn4(self.fc1(flat)))
    xb = F.relu(self.bn5(self.fc2(xb)))
    
    #initialize as identity
    init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
    if xb.is_cuda:
      init=init.cuda()
    matrix = self.fc3(xb).view(-1,self.k,self.k) + init
    return matrix
