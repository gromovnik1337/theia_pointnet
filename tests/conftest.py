import pytest
import pathlib


@pytest.fixture
def test_dir():
    return pathlib.Path(__file__).absolute().parent / 'test_data'
