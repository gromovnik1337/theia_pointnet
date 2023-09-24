from typing import Union
from typing import List
from typing import Tuple
import argparse
import time
import pathlib
import gc
import torch
import tqdm
import numpy as np
from config import config
from data_processing.model.model import PointNetClassification
from data_processing.model.model import point_net_loss
import data_processing.model.dataset as dataset
from torchvision import transforms


def save_model_and_loss(
    output_dir: Union[pathlib.Path, str],
    model: torch.nn.Module,
    model_name: str,
    train_loss: List[float],
    valid_loss: List[float],
) -> None:
    """Saves the trained model & its loss function values in each training step.

    Args:
        output_dir: Absolute path to the output folder.
        model: Trained model.
        model_name: Name of the trained model.
        train_loss: Loss data generated during training.
        valid_loss: Loss data generated during validation.
    """
    output_dir = pathlib.Path(output_dir)
    if not output_dir.exists:
        pathlib.mkdir(output_dir)

    torch.save(model, output_dir / (model_name + ".pt"))
    torch.save(model.state_dict(), output_dir / (model_name + "_state_dict"))

    with open(output_dir / (model_name + "_training_loss.txt"), "w") as f:
        for element in train_loss:
            f.write(str(element) + "\n")
    with open(output_dir / (model_name + "_validation_loss.txt"), "w") as f:
        for element in valid_loss:
            f.write(str(element) + "\n")
    print("Model, state dictionary and loss values saved at: ", str(output_dir))

def train_point_net(
    model: torch.nn.Module,
    device: torch.device,
    epochs: int,
    train_loader: torch.utils.data.dataloader.DataLoader,
    valid_loader: torch.utils.data.dataloader.DataLoader,
    alpha: float,
) -> Tuple[torch.nn.Module, List[float], List[float]]:
    """Performs training of the PointNet input model.

    Args:
        model: Model that is to be trained.
        device: PyTorch device (gpu or cpu) aka. hardware that will be used for computation.
        epochs: Number of training epochs.
        train_loader: PyTorch DataLoader object, iterator through training dataset.
        valid_loader: PyTorch DataLoader object, iterator through validation dataset.
        alpha: Regularization weight of the PointNet loss function.

    Returns:
        Trained model and loss function values.
    """
    gc.collect()  # Garbage colection.
    train_losses_all = []
    valid_losses_all = []
    time_start = time.time()
    print("Training the model!")

    for epoch in tqdm.trange(epochs):
        # Training loop.
        model.train()
        train_loss = 0.0
        for data in train_loader:
            inputs = data["pc"].to(device).float()
            labels = data["category_idx"].to(device)
            # Clear the gradients.
            optimizer = model.optimizer
            optimizer.zero_grad()
            # Forward pass.
            outputs, input_transform, feature_transform = model(inputs.transpose(1, 2))
            # Compute the loss.
            loss = point_net_loss(
                outputs, labels, input_transform, feature_transform, alpha
            )
            # Backpropagate the loss & compute gradients.
            loss.backward()
            # Update weights.
            optimizer.step()

            train_loss += loss.item()

        train_loss_in_epoch = train_loss / len(train_loader)

        # Validation loop.
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():  # Turn off the gradients for validation.
            for data in valid_loader:
                inputs = data["pc"].to(device).float()
                labels = data["category_idx"].to(device)
                # Forward pass.
                outputs, input_transform, feature_transform = model(
                    inputs.transpose(1, 2)
                )
                # Compute the loss.
                loss = point_net_loss(
                    outputs, labels, input_transform, feature_transform
                )

                valid_loss += loss.item()

        valid_loss_in_epoch = valid_loss / len(valid_loader)

        print("Epoch", epoch + 1, "complete!"),
        print("\tTraining Loss: ", round(train_loss_in_epoch, 4))
        print("\tValidation Loss: ", round(valid_loss_in_epoch, 4))
        train_losses_all.append(train_loss_in_epoch)
        valid_losses_all.append(valid_loss_in_epoch)

    time_end = time.time()
    print("Training finished!")
    print(f"Total training time: {int((time_end - time_start) / 60)} min.")

    return model, train_losses_all, valid_losses_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PointNet Classification Training - MCB dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model that is to be trained.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Absolute path of the output directory.",
    )

    args = parser.parse_args()

    # Set the device & clean the memory
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    torch.cuda.empty_cache()

    # Load global config.
    config_file = config.Config()

    # Set the seeds:
    seed = config_file.config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get and set required dataset parameters.
    dataset_train_path = pathlib.Path(config_file.config["dataset"]["train"])
    dataset_test_path = pathlib.Path(config_file.config["dataset"]["test"])
    train_transforms = transforms.Compose(
        [dataset.NormalizePc(), dataset.ApplyRandomRotationZ(), dataset.AddJitter()]
    )
    valid_transforms = transforms.Compose([dataset.NormalizePc()])
    batch_size = config_file.config["batch_size"]

    print("Loading training and testing (validation) dataset!")
    dataset_train = dataset.McbData(dataset_train_path, train_transforms)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=4
    )

    dataset_valid = dataset.McbData(dataset_train_path, valid_transforms)
    valid_loader = torch.utils.data.DataLoader(
        dataset=dataset_valid, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Instantiate a model.
    learning_rate = config_file.config["lr"]
    point_net = PointNetClassification(len(dataset_train.classes), learning_rate)
    point_net.to(device)

    # Train the model.
    epochs = config_file.config["epochs"]
    alpha = config_file.config["alpha"]
    trained_model, train_loss, valid_loss = train_point_net(
        model=point_net,
        device=device,
        epochs=epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        alpha=alpha,
    )

    save_model_and_loss(
        args.output_path, point_net, args.model_name, train_loss, valid_loss
    )
