"""Handles dataset creation for the PyTorch models.

References:
    https://pytorch.org/docs/stable/data.html
    https://github.com/aladdinpersson/Machine-\
    Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset/custom_dataset.py
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://shashikachamod4u.medium.com/excel-csv-to-pytorch-dataset-def496b6bcc1
"""
from config import config
from data_processing.utils import viewer
import pathlib
import trimesh
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Union
from typing import Type
from typing import Dict

SEED = 42

class SamplePc(object):
    def __init__(self, points: int = 1024):
        """
        Args:
            points: Number of points to sample from the mesh.
        """
        self.points = points
    def __call__(self, mesh: trimesh.Trimesh):
        """Samples the point cloud from the mesh surface.
        
        Args:
            mesh: Input mesh.
        Returns:
            Sampled point cloud.
        """
        pc_sampled, _ = trimesh.sample.sample_surface(mesh, self.points, seed=SEED)
    
        return pc_sampled
    

class NormalizePc(object):
    def __call__(self, pc:np.ndarray)-> np.ndarray:
        # TODO(vice) Check the paper for name of the normalization
        """Normalizes point cloud.
        
        Args:
            pc: Input point cloud.
        
        Returns:
            Normalized point cloud.
        """
        if len(pc.shape) != 2:
            print('Invalid point cloud!')
            return np.array([])
        pc_norm = pc - np.mean(pc, axis=0)
        pc_norm /= np.max(np.linalg.norm(pc_norm, axis=1))
        
        return pc_norm
    
class ApplyRandomRotationZ(object):
    def __call__(self, pc:np.ndarray) -> np.ndarray:
        """Applies random rotation around the z axis
        to the input point cloud.
    
        Args:
            pc: Input point cloud.
    
        Returns:
            Rotated point cloud.
        """
        theta = np.random.random(1) * 2 * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                               [math.sin(theta), math.cos(theta), 0],
                               [0, 0, 1]])
          
        pc_rot = rot_matrix.dot(pc.T).T
    
        return pc_rot
    
class AddJitter(object):
    def __call__(self, pc:np.ndarray) -> np.ndarray:
        """Applied random jitter to the point cloud.
    
        Args:
            pc: Input point cloud.
    
        Returns:
            Point cloud with added noise.
        """
        jitter = np.random.normal(0, 0.02, (pc.shape))
        pc_noisy = pc + jitter
        
        return pc_noisy
    
DEFAULT_TRANSFORMS = transforms.Compose([
                                NormalizePc()
                               ])


class McbData(Dataset):
    def __init__(self, dataset_dir: Union[pathlib.Path, str],
                 transforms: Type[transforms.transforms.Compose] = DEFAULT_TRANSFORMS):
        """Performs initial loading of the MCB dataset, as a collection
        of mesh paths and their categories. Also, it loads the transformations
        that are to be applied on each data entry.
        
        Args:
            dataset_dir: Input directory.
            transforms: A composition of transformations that are to be applied
                        on the input point cloud.
        """
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.transforms = transforms
        
        # Create a dict with class names & their indices.
        folders = [dir.stem for dir in sorted(dataset_dir.iterdir()) if dir.is_dir()]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        # Load all the samples paths and their category idx.
        self.samples = []
        for category, category_idx in self.classes.items():
            cat_dir = self.dataset_dir/pathlib.Path(category)
            for mesh in cat_dir.iterdir():
                if mesh.is_file() and mesh.suffix == '.obj':
                    sample = {}
                    sample['mesh_path'] = mesh.absolute()
                    sample['category_idx'] = category_idx
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __preproc__(self, pc: np.ndarray) -> np.ndarray:
        if self.transforms:
            pc = self.transforms(pc)
        return pc

    def __getitem__(self, idx: int) -> Dict[torch.tensor, int]:
        """Returns a single entry of the dataset which consists of input point
        cloud and the output category.
    
        Args:
            idx: Index of the entry.
    
        Returns:
            Tuple containing dataset entry.
        """
        # Load the mesh & sample the point cloud out of it.
        mesh_path = self.samples[idx]['mesh_path']
        category_idx = self.samples[idx]['category_idx']
        mesh = trimesh.load(mesh_path, force='mesh')
        pc = SamplePc(1024)(mesh) 
        # Apply transformations:
        pc = self.__preproc__(pc)
        # Create Pytorch tensor output.
        pc = torch.from_numpy(pc)
        return {'pc': pc, 
               'category_idx': category_idx}