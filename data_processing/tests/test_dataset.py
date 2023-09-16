from data_processing.model import dataset
from torchvision import transforms

def test_dataset(test_dir):
    # Load the test dataset.
    ds = dataset.McbData(test_dir)
    assert len(ds) == 1
    pc = ds[0]['pc']
    # Ensure that the default PointNet sampling has been used.
    assert pc.shape[0] == 1024
    # Test the transformations
    test_transforms = transforms.Compose([
                              dataset.NormalizePc(),
                              dataset.ApplyRandomRotationZ(),
                              dataset.AddJitter()
                            ])
    ds = dataset.McbData(test_dir, test_transforms)
    pc_transformed = ds[0]['pc']
    assert pc.sum() != pc_transformed.sum()


