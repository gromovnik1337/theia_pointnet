from data_processing.model import dataset
from torchvision import transforms
import torch


def test_dataset(test_dir):
    ds = dataset.McbData(test_dir)
    assert len(ds) == 2
    pc = ds[0]["pc"]
    # Ensure that the default PointNet sampling has been used.
    assert pc.shape[0] == 1024
    # Test the transformations
    test_transforms = transforms.Compose(
        [dataset.NormalizePc(), dataset.ApplyRandomRotationZ(), dataset.AddJitter()]
    )
    ds = dataset.McbData(test_dir, test_transforms)
    pc_transformed = ds[0]["pc"]
    assert pc.sum() != pc_transformed.sum()


def test_dataset_loaded(test_dir):
    # Load the test dataset.
    ds = dataset.McbData(test_dir)
    ds_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=2, shuffle=True)
    assert len(ds_loader) == 1
    # Test proper category indexing.
    test_category_idx = torch.tensor([0])
    for el in ds_loader:
        assert torch.eq(el["category_idx"][0], test_category_idx)
