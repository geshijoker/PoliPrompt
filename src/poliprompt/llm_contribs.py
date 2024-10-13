import numpy as np

import openai
import voyageai
import anthropic

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.language_models.chat_models import BaseChatModel

from ratelimit import limits, sleep_and_retry
from typing import List, Dict

def create_llm(llm_name, model_config: Dict) -> BaseChatModel:
    if "claude" in llm_name:
        model = ChatAnthropic(
            model=llm_name, max_tokens=model_config["max_tokens"], temperature=model_config["temperature"]
        )
    elif "gpt" in llm_name:
        model = ChatOpenAI(
            model_name=llm_name, max_tokens=model_config["max_tokens"], temperature=model_config["temperature"]
        )
    elif "mistral" in llm_name:
        model = ChatMistralAI(
            model=llm_name, max_tokens=model_config["max_tokens"], temperature=model_config["temperature"]
        )
    else:
        raise ValueError(f"The specified model {llm_name} is not supported.")
    return model


# LLM API direct with exponential backoff
def call_llm_wrapper(calls, period):
    @sleep_and_retry
    @limits(calls=calls, period=period)
    def call_llm(chain, input_pairs: Dict):
        response = chain.invoke(input_pairs)
        return response

    return call_llm


def get_llm_embeddings(service, docs, batch_size, model):
    """
    Fetches embeddings for a batch of texts using the voyageai model.

    Parameters:
        - docs (list of str): A list of docs to embed.
        - batch_size (int): The number of texts to process in each batch.
        - model: The embedding model to use.

    Returns:
        - numpy array of vectors of float: The embeddings for the input docs.
    """
    if service == "openai":
        embeddings = get_openai_embeddings(docs, batch_size, model)
    elif service == "voyageai":
        embeddings = get_voyageai_embeddings(docs, batch_size, model)
    else:
        raise ValueError(f"Unsupported service: {service}. Choose 'voyageai' or 'openai'.")
    return embeddings


def get_voyageai_embeddings(docs, batch_size, model="voyage-2"):
    """
    Fetches embeddings for a batch of texts using the voyageai model.

    Parameters:
        - docs (list of str): A list of docs to embed.
        - batch_size (int): The number of texts to process in each batch.
        - model: The embedding model to use.

    Returns:
        - numpy array of vectors of float: The embeddings for the input docs.
    """

    client = voyageai.Client()
    embeddings = []

    # Process the docs in batches
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_ebs = client.embed(batch_docs, model=model)
        embeddings.extend(batch_ebs.embeddings)
    return np.array(embeddings)


def get_openai_embeddings(docs, batch_size, model="text-embedding-3-small"):
    """
    Fetches embeddings for a batch of texts using the openai model.

    Parameters:
        - docs (list of str): A list of docs to embed.
        - batch_size (int): The number of texts to process in each batch.
        - model: The embedding model to use.

    Returns:
        - numpy array of vectors of float: The embeddings for the input docs.
    """

    client = openai.OpenAI()
    embeddings = []

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_ebs = client.embeddings.create(input=batch_docs, model=model).data
        embeddings.extend([eb.embedding for eb in batch_ebs])
    return np.array(embeddings)
