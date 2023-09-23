import torch
import numpy as np
from data_processing.model import dataset
from torchvision import transforms
from data_processing.model.model import PointNetClassification
from data_processing.model.train_and_save import train_point_net

_LOSS_TEST_TOL = 0.1


def test_model(test_dir):
    ds_train = dataset.McbData(test_dir)
    # Instantiate a model.
    learning_rate = 0.001
    point_net = PointNetClassification(len(ds_train.classes), learning_rate)
    assert point_net.lr == learning_rate
    assert point_net.n_classes == 1


def test_training(test_dir):
    # Load the training and test datasets.
    train_transforms = transforms.Compose(
        [dataset.NormalizePc(), dataset.ApplyRandomRotationZ(), dataset.AddJitter()]
    )
    valid_transforms = transforms.Compose([dataset.NormalizePc()])
    ds_train = dataset.McbData(test_dir, train_transforms)
    ds_valid = dataset.McbData(test_dir, valid_transforms)
    assert len(ds_train) == 2 == len(ds_valid)

    train_loader = torch.utils.data.DataLoader(dataset=ds_train, batch_size=2, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=ds_valid, batch_size=2, shuffle=True)

    # Instantiate a model.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.001
    point_net = PointNetClassification(len(ds_train.classes), learning_rate)
    point_net.to(device)

    epochs = 1
    _, train_loss, valid_loss = train_point_net(
        model=point_net,
        device=device,
        epochs=epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        alpha=0.0001,
    )
    # Test trivial training loss.
    assert np.allclose(train_loss, np.array([0.0]), _LOSS_TEST_TOL, _LOSS_TEST_TOL)
    assert np.allclose(valid_loss, np.array([0.0]), _LOSS_TEST_TOL, _LOSS_TEST_TOL)