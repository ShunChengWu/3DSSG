# -*- coding: utf-8 -*-
import h5py
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def _load_mat_file(filepath, sequence_key, targets_key=None):
    """
    Loads data from a `*.mat` file or a `*.h5` file.
    Parameters
    ----------
    filepath : str
        The path to the file to load the data from.
    sequence_key : str
        The key for the sequences data matrix.
    targets_key : str, optional
        Default is None. The key for the targets data matrix.
    Returns
    -------
    (sequences, targets, h5py_filehandle) : \
            tuple(array-like, array-like, h5py.File)
        If the matrix files can be loaded with `scipy.io`,
        the tuple will only be (sequences, targets). Otherwise,
        the 2 matrices and the h5py file handle are returned.
    """
    try:  # see if we can load the file using scipy first
        mat = sio.loadmat(filepath)
        targets = None
        if targets_key:
            targets = mat[targets_key]
        return (mat[sequence_key], targets)
    except (NotImplementedError, ValueError):
        mat = h5py.File(filepath, 'r')
        sequences = mat[sequence_key]
        targets = None
        if targets_key:
            targets = mat[targets_key]
        return (sequences, targets, mat)


class MatDataset(Dataset):
    def __init__(self, filepath, split='train'):
        super().__init__()
        filepath = filepath + "/{}.mat".format(split)
        sequence_key = "{}xdata".format(split)
        targets_key = "{}data".format(split)
        out = _load_mat_file(filepath, sequence_key, targets_key)
        self.data_tensor = out[0]
        self.target_tensor = out[1]
        self.split = split

    def __getitem__(self, index):
        if self.split == "train":
            data_tensor = self.data_tensor[:, :, index]
            data_tensor = data_tensor.transpose().astype('float32')
            target_tensor = self.target_tensor[:, index].astype('float32')
        else:
            data_tensor = self.data_tensor[index].astype('float32')
            target_tensor = self.target_tensor[index].astype('float32')
        data_tensor = torch.from_numpy(data_tensor)
        target_tensor = torch.from_numpy(target_tensor)
        return data_tensor, target_tensor

    def __len__(self):
        if self.split == 'train':
            return self.target_tensor.shape[1]
        return self.target_tensor.shape[0]
