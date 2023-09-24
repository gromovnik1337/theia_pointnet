"""Contains all the building blocks of PointNet Classification model.

Source: Qi, Charles R. et. al. - PointNet Deep Learning on Point Sets for
3D Classification and Segmentation
"""
import torch
import torch.nn.functional as F


class Tnet(torch.nn.Module):
    """T-net model, designed to predict transformation to the canonical orientation for each
    input point cloud or the input feature vector. It aims to achieve that final predicitons are
    invariant under transformation. Architecture resembles larger PointNet.
    """

    def __init__(self, k: int = 3):
        """
        Args:
            k: Defines the input vector dimension.
        """
        super().__init__()
        self.k = k
        # Define convolutional layers.
        self.conv1 = torch.nn.Conv1d(self.k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # Define fully connected layers (MLP).
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, self.k * self.k)
        # Define batch normalizations.
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Defines 1 forward pass of the T-net.
        Output transformation is generated from the elements of the last vector, which is the
        output of the last fully connected layer.

        Args:
            X: Input data.

        Returns:
            Predicted transformation.
        """
        # Pass the input through conv layers with ReLU activation.
        xb = F.relu(self.bn1(self.conv1(X)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        # Apply the symmetrical function.
        pool = torch.nn.MaxPool1d(xb.size(-1))(xb)
        # Pass through fully connected layers.
        flat = torch.nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        # Initialize transformation matrix as identity matrix.
        batch_size = X.size(0)
        init = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1)
        if xb.is_cuda:
            init = init.cuda()

        # Create the output transformation from the output of the last fully connected layer.
        xb = self.fc3(xb)
        transformation = xb.view(-1, self.k, self.k) + init

        return transformation


class Transform(torch.nn.Module):
    """Defines first portion of the PointNet architecture.
    Wraps the inference of the two transformations matrices, one for the input point cloud and
    one for the feature vector.

    N.B. Output feature vector and transformation matrices are given separatelt in the output
    as they are all required to compute the loss.
    """

    def __init__(self):
        super().__init__()
        self.t_net_input = Tnet(k=3)
        self.t_net_feature = Tnet(k=64)

        # Define convolutional layers.
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # Define batch normalizations.
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Defines 1 forward pass of the wrapped transformation network.

        Args:
            X: Input data.

        Returns:
            Global feature vector and transformation matrices.
        """
        # Transform the input point cloud using batch matrix multiplication.
        input_transform = self.t_net_input(X)
        xb = torch.bmm(torch.transpose(X, 1, 2), input_transform).transpose(1, 2)

        # Generate feature vector of the transformed cloud.
        xb = F.relu(self.bn1(self.conv1(xb)))

        # Transform the feature vector.
        feature_transform = self.t_net_feature(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), feature_transform).transpose(1, 2)

        # Expand the feature vector & apply symmetric function.
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = torch.nn.MaxPool1d(xb.size(-1))(xb)
        # Generate global feature vector.
        output = torch.nn.Flatten(1)(xb)

        return output, input_transform, feature_transform


class PointNetClassification(torch.nn.Module):
    """PointNet Classification model.
    It extends on the output from the Transform network, adding fully connected layers that
    produce final output scores (which are given as probabilities).
    """

    def __init__(self, n_classes: int, lr: float = 0.001):
        """
        Args:
            n_classes: Number of classes that span possible predictions.
            lr: Learning rate of the model's Adam optimizer.
        """
        super().__init__()
        self.transform = Transform()
        self.n_classes = n_classes
        self.lr = lr
        # Define fully connected layers.
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, self.n_classes)
        # Define batch normalizations.
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)

        # Define droput rate & logarthmic softmax transformation
        # to achieve output whose elements lie in range [0, 1] and
        # sum up to 1.
        self.dropout = torch.nn.Dropout(p=0.3)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Define optimizer.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Defines 1 forward pass of the PointNet classification
        network. Last layer includes dropout!

        N.B. At this point, transformation matrices are not changed.

        Args:
            X: Input data.

        Returns:
            Global feature vector and transformation matrices.
        """
        xb, input_transform, feature_transform = self.transform(X)
        # Pass the input through fully connected layers with ReLU activation.
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        outputs = self.fc3(xb)

        return self.logsoftmax(outputs), input_transform, feature_transform


def point_net_loss(
    outputs: torch.tensor,
    labels: torch.tensor,
    input_transform: torch.tensor,
    feature_transform: torch.tensor,
    alpha: float = 0.0001,
) -> torch.tensor:
    """Loss computation of the PointNet Classification model. Loss is based on the negative
    likelihood loss computation, which is common for classificiation problems. Regularization
    term is added to it, which constrains transformation matrices to be close to orthogonal
    matrix - to minimize the difficulty of their optimization.

    Args:
        outputs: Predicted labels.
        labels: Ground truth labels.
        input_transform: Input point cloud transformation.
        feature_transform: Feature vector transformation.
        alpha: Regularization term weight.

    Returns:
        Compute loss.
    """
    criterion = torch.nn.NLLLoss()  # Negative likelihood loss.

    # Initialize identity transformations.
    batch_size = outputs.size(0)
    id_input_transform = torch.eye(3, requires_grad=True).repeat(batch_size, 1, 1)
    id_feature_transform = torch.eye(64, requires_grad=True).repeat(batch_size, 1, 1)

    if outputs.is_cuda:
        id_input_transform = id_input_transform.cuda()
        id_feature_transform = id_feature_transform.cuda()

    # Compute transformation regularization term (Lreg from the paper).
    l_reg_input = id_input_transform - torch.bmm(
        input_transform, input_transform.transpose(1, 2)
    )
    l_reg_feature = id_feature_transform - torch.bmm(
        feature_transform, feature_transform.transpose(1, 2)
    )

    loss = criterion(outputs, labels) + alpha * (
        torch.norm(l_reg_input) + torch.norm(l_reg_feature)
    ) / float(batch_size)

    return loss
