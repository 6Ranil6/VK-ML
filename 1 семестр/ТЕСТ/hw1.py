import numpy as np


def product_of_diagonal_elements_vectorized(matrix: np.array):
    mask = np.diag(matrix) != 0
    return np.diag(matrix)[mask].prod()


def are_equal_multisets_vectorized(x: np.array, y: np.array):
    return np.all(np.sort(x) == np.sort(y))


def max_before_zero_vectorized(x: np.array):
    zero_index = None
    if x[x.size - 1] == 0:
        zero_index = np.where(x[:-2] == 0)[0]
    else:
        zero_index = np.where(x == 0)[0]
    return x[zero_index + 1].max()


def add_weighted_channels_vectorized(image: np.array):
    weights = np.array([0.299, 0.587, 0.114])
    grayscale_image = np.tensordot(image, weights, axes=([2], [0]))
    return grayscale_image


def run_length_encoding_vectorized(x: np.array):
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])
    diff = np.diff(x)
    indices = np.where(diff != 0)[0] + 1
    indices = np.concatenate(([0], indices, [n]))
    values = x[indices[:-1]]
    lengths = np.diff(indices)
    return values, lengths
