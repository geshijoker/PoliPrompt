from scipy.spatial.distance import cdist
from typing import List, Dict
import pandas as pd
import numpy as np

def maximal_marginal_relevance(
    query_embedding, pool_embeddings, selected_indices, k, lambda_param=1.0, metric="euclidean"
):
    """
    Apply the Maximal Marginal Relevance (MMR) algorithm to select k items.

    Parameters:
        - query_embedding (np.ndarray): The embedding of the query item.
        - pool_embeddings (np.ndarray): The embeddings of the items in the pool.
        - selected_indices (list of str): The preselected indices.
        - k (int): The number of items to select.
        - lambda_param (float): The trade-off parameter between relevance and diversity (0 <= lambda_param <= 1).

    Returns:
        - selected_indices (list): The indices of the selected items in the pool.
    """

    # Calculate the distances between the query and the pool embeddings
    relevance_scores = cdist([query_embedding], pool_embeddings, metric=metric).flatten()

    # List to store the indices of selected items
    if not selected_indices:
        selected_indices = []

    # While we have not yet selected k items
    for _ in range(k):
        if not selected_indices:
            # Select the item with the highest relevance (smallest distance)
            selected_idx = int(np.argmin(relevance_scores))
        else:
            # Calculate diversity for each unselected item
            diversity_scores = np.min(cdist(pool_embeddings[selected_indices], pool_embeddings, metric=metric), axis=0)
            # Combine relevance and diversity to score items
            mmr_scores = lambda_param * relevance_scores - (1 - lambda_param) * diversity_scores
            # Select the item with the highest MMR score
            selected_idx = int(np.argmax(mmr_scores))

        # Append the selected index to the list of selected indices
        selected_indices.append(selected_idx)

        # Mark the selected item as used by setting its score to -infinity
        relevance_scores[selected_idx] = -np.inf

    return selected_indices


def get_k_nearest_distinct_classes(query_embedding, pool_embeddings, pool_labels, kshots, options, metric="euclidean"):
    """
    Get the indices of k embeddings from distinct classes that are most similar to the query_embedding.

    Parameters:
        - query_embedding (np.ndarray): The embedding of the query item.
        - pool_embeddings (np.ndarray): The embeddings of the items in the pool.
        - pool_labels (list or np.ndarray): The class labels corresponding to the items in the pool.
        - kshots (int): The number of distinct classes to select.
        - options (list of str): The classes to select from.
        - metric (str): The distance metric to use (default is 'euclidean').

    Returns:
        - selected_indices (list): The indices of the selected items in the pool.
    """

    # Calculate the distances between the query and the pool embeddings
    distances = cdist([query_embedding], pool_embeddings, metric=metric).flatten()

    # Get the sorted indices based on distance (smallest to largest)
    sorted_indices = np.argsort(distances)

    # Initialize a list to store selected indices and a set for the selected classes
    selected_indices = []
    un_selected_classes = set(options)
    k_selected = 0

    # Iterate through the sorted indices and select distinct classes
    for idx in sorted_indices:
        if pool_labels[idx] in un_selected_classes:
            selected_indices.append(int(idx))
            un_selected_classes.remove(pool_labels[idx])
            k_selected += 1

        # Stop when we've selected k distinct classes
        if len(un_selected_classes) == 0 or k_selected == kshots:
            break

    return selected_indices


def select_kshots(
    ds: pd.DataFrame,
    feature_col: str,
    answer_col: str,
    kshots: int,
    idx: int,
    indices: List[int],
    embeddings: np.ndarray,
    lambda_param: float = 1.0,
    options=None,
) -> List[Dict[str, str]]:
    """
    Select k-shots examples using Maximal Marginal Relevance (MMR) based on Euclidean distance.

    Parameters:
        - ds: datasets.Dataset - The dataset object containing the data.
        - feature_col: str - The column name for the features.
        - answer_col: str - The column name for the answers.
        - kshots: int - The number of shots to select.
        - idx: int - The index of the query example in the dataset.
        - indices: List[int] - The list of indices from which to select the examples.
        - embeddings: np.ndarray - The numpy array containing embeddings of all examples.
        - lambda_param (float): The trade-off parameter between relevance and diversity (0 <= lambda_param <= 1).

    Returns:
        - List[Dict[str, str]]: A list of dictionaries with 'Text' and 'Answer' for the selected k-shots.
    """
    # Extract the embedding for the query example
    query_embedding = embeddings[idx]

    # Extract the embeddings for the pool of examples
    pool_embeddings = embeddings[indices]
    pool_labels = ds.iloc[indices][answer_col].tolist()

    # Initialize the list for selected indices
    selected_indices = []
    if options:
        selected_indices = get_k_nearest_distinct_classes(
            query_embedding, pool_embeddings, pool_labels, kshots, options
        )

    # Perform Maximal Marginal Relevance (MMR) selection
    selected_indices = maximal_marginal_relevance(
        query_embedding, pool_embeddings, selected_indices, kshots - len(selected_indices), lambda_param=lambda_param
    )

    # Get the actual indices from the 'indices' list
    kshots_indices = [indices[i] for i in selected_indices]

    # Extract features and answers for the selected indices
    selected_data = []
    for idx in kshots_indices:
        feature = ds.loc[idx, feature_col]
        answer = ds.loc[idx, answer_col]
        selected_data.append({"content": feature, "answer": answer})

    return selected_data
