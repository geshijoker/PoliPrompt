from abc import ABC, abstractmethod

import numpy as np
import umap
from sklearn.decomposition import PCA


class DimensionalityReducer(ABC):
    """
    Abstract base class for dimensionality reduction techniques.
    """

    @abstractmethod
    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Abstract method to reduce the dimensionality of embeddings.

        Parameters:
            - embeddings (np.ndarray): The high-dimensional data to be reduced.

        Returns:
            - np.ndarray: The reduced-dimensional embeddings.
        """
        pass


class UMAPReducer(DimensionalityReducer):
    def __init__(self, n_components: int = 2, **kwargs):
        """
        Initialize UMAP reducer with specified number of components and other configurations.

        Parameters:
            - n_components (int): The number of dimensions to reduce to.
            - **kwargs: Additional keyword arguments for UMAP configuration.
        """
        self.reducer = umap.UMAP(n_components=n_components, **kwargs)

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:

        # Fit and transform the embeddings
        reduced_embeddings = self.reducer.fit_transform(embeddings)

        return reduced_embeddings


class PCAReducer(DimensionalityReducer):
    def __init__(self, n_components: int = 2, **kwargs):
        """
        Initialize PCA reducer with specified number of components and other configurations.

        Parameters:
            - n_components (int): The number of dimensions to reduce to.
            - **kwargs: Additional keyword arguments for PCA configuration.
        """
        self.reducer = PCA(n_components=n_components, **kwargs)

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce the dimensionality of the embeddings using PCA.

        Parameters:
            - embeddings (np.ndarray): The high-dimensional data to be reduced.

        Returns:
            - np.ndarray: The reduced-dimensional embeddings.
        """
        return self.reducer.fit_transform(embeddings)


def create_reducer(method: str, n_components: int = 2, **kwargs) -> DimensionalityReducer:
    """
    Factory method to create the appropriate reducer based on the method name.

    Parameters:
        - method (str): The dimensionality reduction method ('umap' or 'pca').
        - n_components (int): The number of components for the dimensionality reduction.
        - **kwargs: Additional keyword arguments for the reducer configuration.

    Returns:
        - DimensionalityReducer: An instance of a reducer.
    """
    reducers = {
        "umap": UMAPReducer,
        "pca": PCAReducer,
    }

    if method.lower() not in reducers:
        raise ValueError(f"Unsupported method: {method}. Supported methods are: {list(reducers.keys())}.")

    return reducers[method.lower()](n_components=n_components, **kwargs)
