from setuptools import find_packages, setup

setup(
    name="theia_pointnet",
    version="1.0.0",
    packages=find_packages(include=["config", "model", "notebooks", "tests", "utils"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy==1.8.1",
        "matplotlib",
        "plotly",
        "pytest",
        "notebook",
        "ipykernel",
        "pandas",
        "torch",
        "torchvision",
        "scikit-learn",
        "trimesh",
        "black",
        "tqdm",
    ],
)
