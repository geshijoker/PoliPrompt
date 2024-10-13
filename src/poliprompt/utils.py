import os
import csv
import json
from typing import List, Dict
import pandas as pd
from datetime import datetime as dt
from pathlib import Path

def ensure_workstation_directories(work_station: Path):
    """
    Ensures that the WORK_STATION directory and its subdirectories (infiles/ and outfiles/) exist.
    """
    if not work_station.parent.exists():
        raise FileNotFoundError(f"No valid parent directories exist for the path '{work_station}'.")

    # Paths to infiles/ and outfiles/ directories
    infiles_dir = work_station / "infiles"
    configs_dir = infiles_dir / "configs"
    prompts_dir = infiles_dir / "prompts"
    outfiles_dir = work_station / "outfiles"
    logs_dir = outfiles_dir / "logs"

    # Create directories if they do not exist
    if not work_station.exists():
        work_station.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {infiles_dir}")

    if not infiles_dir.exists():
        infiles_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {infiles_dir}")

    if not configs_dir.exists():
        configs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {configs_dir}")

    if not prompts_dir.exists():
        prompts_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {prompts_dir}")

    if not outfiles_dir.exists():
        outfiles_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {outfiles_dir}")

    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {logs_dir}")


def validate_csv_file(file_path: str, feature_col: str, answer_col: str):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    # Check if the file is a CSV
    if not file_path.endswith(".csv"):
        raise ValueError(f"File '{file_path}' is not a CSV file.")

    # Load the CSV file using pandas
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read the CSV file '{file_path}': {e}")

    # Check if feature_col exists in the DataFrame
    if feature_col not in df.columns:
        raise ValueError(f"Feature column '{feature_col}' does not exist in the CSV file.")

    # Check if answer_col exists in the DataFrame
    if answer_col not in df.columns:
        raise ValueError(f"Answer column '{answer_col}' does not exist in the CSV file.")


def read_docs_from_dataframe(df: pd.DataFrame, column_name: str="text"):
    """
    Reads texts from a specified column in a CSV file.

    Parameters:
        - file_path (str): The path to the CSV file.
        - column_name (str): The name of the column containing the texts.

    Returns:
        - list: A list of docs from the specified column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but got {type(df)} instead.")

    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Extract the texts as a list
    docs = df[column_name].tolist()

    return docs


def track_computation_time(elapsed_time: float) -> None:
    """
    Tracks the computation time and prints it in hours, minutes, and seconds.

    Parameters:
        - elapsed_time (float): The elapsed time using time.time().
    """
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return hours, minutes, seconds
    print(f"Computation time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")


def load_config(configs_path: str | Path, name: str) -> Dict:
    """
    Load the configuration from a specific json file.

    Parameters:
    - configs_path (str): Path to the configuration JSON file.
    - name (str): Name of the config to use.

    Returns:
    - dict: The configuration for the specified LLM.
    """
    try:
        with open(configs_path, "r") as file:
            configs = json.load(file)
            config = configs.get(name)
            if config is None:
                raise ValueError(f"No configuration found for: {name}")
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {configs_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from file: {configs_path}")


def get_prompt(prompt_file: str | Path) -> str:
    if isinstance(prompt_file, str):
        prompt_file = Path(prompt_file)
    return prompt_file.read_text()

