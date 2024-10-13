from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import KMeans


class ExemplarSelector(ABC):
    """
    Abstract base class for selecting exemplars from a set of embeddings.
    """

    @abstractmethod
    def select_exemplars(self, embeddings: np.ndarray, n_exemplars: int = 10, **kwargs) -> list:
        """
        Abstract method to select exemplars.

        Parameters:
            - embeddings (numpy.ndarray): The embeddings to select exemplars from.
            - n_exemplars (int): The number of exemplars to select.
            - **kwargs: Additional keyword arguments for exemplar selection.

        Returns:
            - list: Indices of the selected exemplar embeddings.
        """
        pass


class RandomExemplarSelector(ExemplarSelector):
    """
    Exemplar selector that selects random exemplars.
    """
    def __init__(self):
        pass

    def select_exemplars(self, embeddings: np.ndarray, n_exemplars: int = 10, **kwargs) -> list:
        """
        Selects random exemplar embeddings.

        Parameters:
            - embeddings (np.ndarray): The embeddings to select exemplars from.
            - n_exemplars (int): The number of exemplars to select.

        Returns:
            - list: Indices of the randomly selected exemplar embeddings.
        """
        return np.random.choice(len(embeddings), n_exemplars, replace=False).tolist()


class KMeansExemplarSelector(ExemplarSelector):
    """
    Exemplar selector that uses KMeans clustering to select exemplars.
    """
    def __init__(self):
        pass

    def select_exemplars(self, embeddings: np.ndarray, n_exemplars: int = 10, **kwargs) -> list:
        """
        Selects exemplar embeddings using KMeans clustering.

        Parameters:
            - embeddings (numpy.ndarray): The embeddings to select exemplars from.
            - n_exemplars (int): The number of exemplars to select.
            - **kwargs: Additional keyword arguments for KMeans configuration.

        Returns:
            - list: Indices of the selected exemplar embeddings.
        """

        # Initialize KMeans model with additional configurations
        kmeans = KMeans(n_clusters=n_exemplars, **kwargs)

        # Fit the model to the embeddings
        kmeans.fit(embeddings)

        # Select exemplars based on the closest points to the cluster centers
        exemplars_indices = []
        for i in range(n_exemplars):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            cluster_center = kmeans.cluster_centers_[i]
            closest_index = cluster_indices[
                np.argmin(np.linalg.norm(embeddings[cluster_indices] - cluster_center, axis=1))
            ]
            exemplars_indices.append(closest_index)

        return exemplars_indices


def create_selector(method: str) -> ExemplarSelector:
    """
    Factory method to create the appropriate reducer based on the method name.

    Parameters:
        - method (str): The dimensionality reduction method ('umap' or 'pca').
        - n_exemplars (int): The number of exemplars to select.
        - **kwargs: Additional keyword arguments for KMeans configuration.

    Returns:
        - ExemplarSelector: An instance of a selector.
    """

    selectors = {
        "kmeans": KMeansExemplarSelector,
        "random": RandomExemplarSelector,
    }

    if method.lower() not in selectors:
        raise ValueError(f"Unsupported method: {method}. Supported methods are: {list(selectors.keys())}.")

    return selectors[method.lower()]()
