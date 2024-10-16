import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List

import json
import yaml
import numpy as np
from tqdm import tqdm
import pandas as pd

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

from .utils import *
from .llm_contribs import create_llm, call_llm_wrapper, get_llm_embeddings
from .reducers import create_reducer
from .selectors import create_selector
from .retrieves import select_kshots


# Set the logging level to WARNING to ignore INFO and DEBUG logs of LLM requests
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Setup logger
logger = logging.getLogger(__name__)

class TextClassifier:
    def __init__(
            self, 
            data_file: str | Path,
            work_station: str | Path,
            env_path: str | Path,
            options: List[str],
            feature_col: str="text",
            answer_col: str="label",
            random_state: int=42,
            requests_per_period: int=60,
            secondf_per_period: int=60, 
        ):
        """

        """
        # The path to the environment file saving passwordf
        self.data_file = Path(os.path.expanduser(data_file))
        # self.data = validate_csv_file(self.data_file)
        self.options = options
        self.feature_col = feature_col
        self.answer_col = answer_col
        self.random_state = random_state
        self.requests_per_period = requests_per_period
        self.secondf_per_period = secondf_per_period

        # Create the working station
        self.work_station = Path(os.path.expanduser(work_station))
        ensure_workstation_directories(self.work_station)
        self.infiles_dir = self.work_station / "infiles"
        self.configs_dir = self.infiles_dir / "configs"
        self.prompts_dir = self.infiles_dir / "prompts"
        self.outfiles_dir = self.work_station / "outfiles"
        self.logs_dir = self.outfiles_dir / "logs"

        # Load the passwordf of LLMs from environment file
        load_dotenv(Path(os.path.expanduser(env_path)))

    def create_few_shot_pool(
            self, 
            embedding_llm_name: str, 
            reduce_method: str,
            select_method: str,
            testing: bool=False,
            testing_size: int=None,
        ):

        # The path to save numpy array embeddings
        embeddings_file = self.outfiles_dir / "embeddings.npy"
        # The path to save indices as json files
        exemplars_file = self.outfiles_dir / "exemplar_indices.json"

        # The path to load configurations
        logger.warning(f"You are supposed to provide 'embedding_llm_configs.json', 'reduce_configs.json', and 'select_configs.json' in the path {self.configs_dir}")

        embedding_llm_config = load_config(self.configs_dir / "embedding_llm_configs.json", embedding_llm_name)
        reduce_config = load_config(self.configs_dir / "reduce_configs.json", reduce_method)
        if "random_state" in reduce_config:
            reduce_config["random_state"] = self.random_state
        select_config = load_config(self.configs_dir / "select_configs.json", select_method)
        if "random_state" in select_config:
            select_config["random_state"] = self.random_state

        start_time = time.time()

        try:
            data = pd.read_csv(self.data_file)
        except Exception as e:
            raise ValueError(f"Failed to read the CSV file '{self.data_file}': {e}")

        # Read docs dataset
        questions = read_docs_from_dataframe(data, column_name=self.feature_col)
        if testing:
            if testing_size is None:
                raise ValueError("testing_size cannot be None in the testing mode. It must be a valid integer.")
            if len(questions) < testing_size:
                logger.info(f"Provided testing_size ({testing_size}) is greater than the number of questions ({len(questions)}), "
                    f"we will only use the first {len(questions)} questions.")
                testing_size = len(questions)
            questions = questions[: testing_size]

        # Step 1: Embed all docs
        embeddings = get_llm_embeddings(embedding_llm_name, questions, **embedding_llm_config)

        # Step 2: Dimension Reduction with UMAP
        reducer = create_reducer(reduce_method, **reduce_config)
        reduced_embeddings = reducer.reduce(embeddings)
        try:
            np.save(embeddings_file, reduced_embeddings)
            logger.info(f"NumPy array saved to {embeddings_file}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while saving the embeddings: {e}")

        # Step 3: Select exemplar docs
        selector = create_selector(method=select_method)
        exemplar_indices = selector.select_exemplars(reduced_embeddings, **select_config)
        exemplar_indices = [int(idx) for idx in exemplar_indices]

        # Step 4: Save exemplars to work station
        try:
            exemplars_file.write_text(json.dumps(exemplar_indices, indent=4))
            logger.info(f"List of integer indices saved to {exemplars_file}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while saving the exemplar indices: {e}")

        # Track and print the computation time
        end_time = time.time()
        elapsed = end_time - start_time
        hours, minutes, secondf = track_computation_time(elapsed)
        logger.info(f"Computation time: {int(hours):02}:{int(minutes):02}:{int(secondf):02}")
        logger.info(f"You are supposed to provid {self.answer_col} in {self.data_file} for the next steps.")

    def optimize_task_description(
            self,
            llm_name: str,
            prompt_file_name: str,
            disable_progress_bar: bool=False,
        ):
        rules_path = self.outfiles_dir / "rules.json"
        llm_configs = self.configs_dir / "llm_configs.json"
        prompt_file = self.prompts_dir / prompt_file_name

        # Construct large language chat model
        model_config = load_config(llm_configs, llm_name)
        model = create_llm(llm_name=llm_name, model_config=model_config)

        # Load prompt and indices
        prompt = get_prompt(prompt_file=prompt_file)
        exemplars_file = self.outfiles_dir / "exemplar_indices.json"
        indices = json.loads(exemplars_file.read_text())

        # Read docs dataset
        df = pd.read_csv(self.data_file)
        contexts = read_docs_from_dataframe(df, column_name=self.feature_col)
        answers = read_docs_from_dataframe(df, column_name=self.answer_col)
        contexts = [contexts[i] for i in indices]
        answers = [answers[i] for i in indices]

        # Generate reasons for human's annotation of each text and save it in the rules_path
        if not os.path.isfile(rules_path):

            # Ask LLM to justify each answer to a question
            qa_sys_prompt = """You are a helpful AI assistant. \
            You will be provided with a task description, a (Text and Answer) pair. \
            The Text is the contexts relevant to the task, and the Answer is a human's choice to the task given the contexts. \
            Your job is to analyze the contexts and the answer, \
            briefly justify why the choice is appropriate for the given contexts by understanding the underlying logic. \
            Summarize a rule that explains why this option is better compared to other options. Starts with \"The correct option is [HUMAN ANSWER].\"\nTask:\"\"\"\n{task}\"\"\"\n
            """
            qa_sys_prompt = qa_sys_prompt.format(task=prompt)

            results = []
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=(qa_sys_prompt)),
                    HumanMessagePromptTemplate.from_template('Text:"""\n{context}"""\nAnswer:"""\n{answer}"""\n'),
                ]
            )

            chain = qa_prompt | model | StrOutputParser()
            call_llm_with_limits = call_llm_wrapper(self.requests_per_period, self.secondf_per_period)
            for i, (context, answer) in tqdm(enumerate(zip(contexts, answers)), total=len(indices), leave=False):
                result = call_llm_with_limits(chain, {"context": context, "answer": answer})
                results.append(result)
            # Save the extracted rules
            rules_path.write_text(json.dumps(results, indent=4))
        else:
            results = json.loads(rules_path.read_text())

        # Concatenate all rules as a document
        docs = []
        for i, result in tqdm(enumerate(results), len(results), disable=disable_progress_bar):
            doc = Document(page_content=result, metadata={"idx": str(i)})
            docs.append(doc)

        # Ask LLM to summarize the rules with MapReduce
        sum_sys_prompt = """You are give a task description. \nTask:\"\"\"\n{task}\"\"\"\nThe task provides a few options to answer a question. \
        The following texts are the answer traces with rules extracted to summarize why an option is correct to the question.\
        Please briefly summarize the rules for valid existing options. \
        The content can be, for example, the meaning of the options, or the reason why we prefer one option than the other in different situations. \
        This summarization will be used to guide human to answer such questions. \
        Please make the summarizaiton concise and actionable."""
        sum_sys_prompt = sum_sys_prompt.format(task=prompt)
        map_template = """\n\"\"\"{docs}\"\"\"\nCONCISE RULE:"""

        map_prompt = PromptTemplate.from_template(sum_sys_prompt + map_template)
        map_chain = LLMChain(llm=model, prompt=map_prompt) 

        reduce_template = """The following is set of summary of rules to justify a few options:
        {docs}
        Take these and distill it into a final, consolidated summary of the main rules for valid options. 
        CONCISE SUMMARY RULES:"""
        reduce_prompt = PromptTemplate.from_template(sum_sys_prompt + reduce_template)
        reduce_chain = LLMChain(llm=model, prompt=reduce_prompt)

        # Take a list of documents, combine them into a single string, and pass it to an LLMChain
        combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")

        # Combine and iteratively reduce the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=4000,
        )

        # Combine documents by mapping a chain over them, then combine results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )

        output_summary = map_reduce_chain.invoke(docs)
        logger.warning(output_summary["output_text"])

        return output_summary["output_text"]

    def annotate(
            self, 
            llm_name: str,
            prompt_file_name: str,
            kshots: int=0,
            lambda_param: float=1.0,
            disable_progress_bar: bool=False,
            testing: bool=False,
            testing_size: int=None,
        ):
        # Input files
        dataset_name = self.data_file.stem
        prompt_file = self.prompts_dir / prompt_file_name
        prompt_name = prompt_file.stem
        llm_configs = self.configs_dir / "llm_configs.json"
        embeddings_file = self.outfiles_dir / "embeddings.npy"
        embeddings = np.load(embeddings_file)
        exemplars_file = self.outfiles_dir / "exemplar_indices.json"
        indices = json.loads(exemplars_file.read_text())
        if len(indices) < kshots:
            logger.info(f"The number of exemplars in the pool ({len(indices)}) is smaller than the user specified {kshots}-shots, "
                    f"we will only use the first {len(indices)}-shots.")
            kshots = len(indices)

        # Create the file to save responses
        if testing:
            output_file_name = f"{dataset_name}_{llm_name}_{prompt_name}_{kshots}shots_test.csv"
        else:
            output_file_name = f"{dataset_name}_{llm_name}_{prompt_name}_{kshots}shots.csv"
        output_file = self.outfiles_dir / output_file_name

        if os.path.isfile(output_file):
            results = read_docs_from_dataframe(pd.read_csv(output_file, index_col=False), self.answer_col)
            logger.info(f"Saving output to an existing file \n{output_file}")
        else:
            results = []
            logger.info(f"Saving output to a new file \n{output_file}")
        processed_count = max(len(results), 0)

        # Construct large language chat model
        model_config = load_config(llm_configs, llm_name)
        model = create_llm(llm_name=llm_name, model_config=model_config)

        # Prepare the unlabelled data to annotate
        df = pd.read_csv(self.data_file, index_col=False)
        df = df[[self.feature_col, self.answer_col]]
        print("Examples of data\n", df.head(n=min(5, len(df))))
        if testing:
            if testing_size is None:
                raise ValueError("testing_size cannot be None in the testing mode. It must be a valid integer.")
            if len(df) < testing_size:
                logger.info(f"Provided testing_size ({testing_size}) is greater than the number of questions ({len(questions)}), "
                    f"we will only use the first {len(df)} questions.")
                testing_size = len(df)

            df = df[:testing_size]
            indices = [idx for idx in indices if idx < testing_size]

        # Create the prefix and suffix from the prompt 
        prefix = get_prompt(prompt_file=prompt_file)
        if kshots > 0:
            prefix = prefix + f" You're given {kshots} examples for references. " 
            suffix = f"That's all {kshots} examples. "
        suffix = suffix + "Perform the task based on the next given text, choose the correct answer from the options (" + ", ".join(self.options) + ') in a single-choice format with options in \'<\' and \'>\'.\nText: """\n{input}\n"""\nAnswer: '
        call_llm_with_limits = call_llm_wrapper(self.requests_per_period, self.secondf_per_period)

        start_time = time.time()
        # Call LLM to annotate unlabelled instances
        try:
            for i, row in tqdm(enumerate(df[self.feature_col]), total=len(df), initial=processed_count, disable=disable_progress_bar, leave=False):
                if i not in indices:
                    examples = select_kshots(
                        df, self.feature_col, self.answer_col, kshots, i, indices, embeddings, lambda_param, self.options
                    )
                    example_prompt = PromptTemplate(
                        input_variables=["content", "answer"], template='Text: """\n{content}\n"""\nAnswer: {answer}'
                    )

                    prompt = FewShotPromptTemplate(
                        examples=examples,
                        example_prompt=example_prompt,
                        prefix=prefix,
                        suffix=suffix,
                        input_variables=["input"],
                    )
                    chain = prompt | model | StrOutputParser()
                    result = call_llm_with_limits(chain, {"input": row})
                else:
                    result = df.loc[i, self.answer_col]
                results.append(result)
        except Exception as e:  # Naked execpt, blame the your LLM API
            logger.error(f"Error when calling LLM API to annotate texts: {e}")
        finally:
            # Track and print the computation time
            end_time = time.time()
            elapsed = end_time - start_time
            hours, minutes, secondf = track_computation_time(elapsed)
            logger.info(f"Computation time: {int(hours):02}:{int(minutes):02}:{int(secondf):02}")
            if results:
                results_df = pd.DataFrame()
                results_df[self.answer_col] = results
                results_df.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
            else:
                raise Exception("Empty results, please check the log.")

    def cot_mismatch_solver(
            self,
            llm_name: str,
            mismatch_file_name: str,
            cot_prompt_file_name: str,
            kshots: int=0,
            lambda_param: float=1.0,
            disable_progress_bar: bool=False,
            testing: bool=False,
            testing_size: int=None,
        ):
        # Input files
        dataset_name = self.data_file.stem
        cot_prompt_file = self.prompts_dir / cot_prompt_file_name
        prompt_name = cot_prompt_file.stem
        llm_configs = self.configs_dir / "llm_configs.json"
        embeddings_file = self.outfiles_dir / "embeddings.npy"
        embeddings = np.load(embeddings_file)
        exemplars_file = self.outfiles_dir / "exemplar_indices.json"
        indices = json.loads(exemplars_file.read_text())
        if len(indices) < kshots:
            logger.info(f"The number of exemplars in the pool ({len(indices)}) is smaller than the user specified {kshots}-shots, "
                    f"we will only use the first {len(indices)}-shots.")
            kshots = len(indices)
        mismatch_file = self.infiles_dir / mismatch_file_name
        mismatch_name = mismatch_file.stem
        mismatch_indices = json.loads(mismatch_file.read_text())

        # Create the file to save responses
        if testing:
            output_file_name = f"cot_{mismatch_name}_{dataset_name}_{llm_name}_{prompt_name}_{kshots}shots_test.csv"
        else:
            output_file_name = f"cot_{mismatch_name}_{dataset_name}_{llm_name}_{prompt_name}_{kshots}shots.csv"
        output_file = self.outfiles_dir / output_file_name

        # Construct large language chat model
        model_config = load_config(llm_configs, llm_name)
        model = create_llm(llm_name=llm_name, model_config=model_config)

        # Prepare the unlabelled data to annotate
        df = pd.read_csv(self.data_file, index_col=False)
        df = df[[self.feature_col, self.answer_col]]
        if testing:
            if testing_size is None:
                raise ValueError("testing_size cannot be None in the testing mode. It must be a valid integer.")
            if len(df) < testing_size:
                logger.info(f"Provided testing_size ({testing_size}) is greater than the number of questions ({len(questions)}), "
                    f"we will only use the first {len(df)} questions.")
                testing_size = len(df)

            df = df[:testing_size]
            indices = [idx for idx in indices if idx < testing_size]
            mismatch_indices = [idx for idx in mismatch_indices if idx < testing_size]

        # Create the prefix and suffix from the prompt 
        prefix = get_prompt(prompt_file=cot_prompt_file)
        suffix = "Please "
        if kshots > 0:
            prefix = prefix + f" You're given {kshots} examples for references. " 
            suffix = f"That's all {kshots} examples with succinct answers. However, For the next text, you need to first analyze it step by step and provide reasoning. Then "
        suffix = suffix + "perform the task by analyzing the following text step by step and provide your reasoning. Do not exceed 50 words. Finally, based on the reasoning, choose the correct answer from the options (" + ", ".join(self.options) + ') in a single-choice format with options in \'<\' and \'>\'.\nText: """\n{input}\n"""\nAnswer: Let\'s think step by step. '
        call_llm_with_limits = call_llm_wrapper(self.requests_per_period, self.secondf_per_period)

        start_time = time.time()
        # Call LLM to annotate unlabelled instances
        results = []
        processed_ids = []
        try:
            for i in tqdm(mismatch_indices, total=len(mismatch_indices), disable=disable_progress_bar, leave=False, desc="Processing indices"):
                if i not in indices:
                    examples = select_kshots(
                        df, self.feature_col, self.answer_col, kshots, i, indices, embeddings, lambda_param, self.options
                    )
                    example_prompt = PromptTemplate(
                        input_variables=["content", "answer"], template='Text: """\n{content}\n"""\nAnswer: {answer}'
                    )

                    cot_prompt = FewShotPromptTemplate(
                        examples=examples,
                        example_prompt=example_prompt,
                        prefix=prefix,
                        suffix=suffix,
                        input_variables=["input"],
                    )
                    row = df.loc[i, self.feature_col]
                    question = {"input": row}

                    chain = cot_prompt | model | StrOutputParser()
                    result = call_llm_with_limits(chain, question)
                else:
                    result = df.loc[i, self.answer_col]
                results.append(result)
                processed_ids.append(i)

        except Exception as e:  # Naked execpt, blame the your LLM API
            logger.error(f"Error when calling LLM API to annotate texts: {e}")

        finally:
            # Track and print the computation time
            end_time = time.time()
            elapsed = end_time - start_time
            hours, minutes, secondf = track_computation_time(elapsed)
            logger.info(f"Computation time: {int(hours):02}:{int(minutes):02}:{int(secondf):02}")
            if results:
                results_df = pd.DataFrame()
                results_df["mismatch ids"] = processed_ids
                results_df[self.answer_col] = results
                results_df.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
            else:
                raise Exception("Empty results, please check the log.")

    def judge_mismatch_solver(
            self, 
            judge_llm_name: str,
            llm1_name: str,
            llm2_name: str,
            mismatch_file_name: str,
            cot_prompt_file_name: str,
            judge_prompt_file_name: str,
            kshots: int=0,
            lambda_param: float=1.0,
            disable_progress_bar: bool=False,
            testing: bool=False,
            testing_size: int=None,
        ):
        # Input files
        dataset_name = self.data_file.stem
        cot_prompt_file = self.prompts_dir / cot_prompt_file_name
        judge_prompt_file = self.prompts_dir / judge_prompt_file_name
        prompt_name = judge_prompt_file.stem
        llm_configs = self.configs_dir / "llm_configs.json"
        embeddings_file = self.outfiles_dir / "embeddings.npy"
        embeddings = np.load(embeddings_file)
        exemplars_file = self.outfiles_dir / "exemplar_indices.json"
        indices = json.loads(exemplars_file.read_text())
        if len(indices) < kshots:
            logger.info(f"The number of exemplars in the pool ({len(indices)}) is smaller than the user specified {kshots}-shots, "
                    f"we will only use the first {len(indices)}-shots.")
            kshots = len(indices)
        mismatch_file = self.infiles_dir / mismatch_file_name
        mismatch_name = mismatch_file.stem
        mismatch_indices = json.loads(mismatch_file.read_text())

        # Create the file to save responses
        if testing:
            output_file_name = f"judge_{mismatch_name}_{dataset_name}_{judge_llm_name}_{prompt_name}_{kshots}shots_test.csv"
        else:
            output_file_name = f"judge_{mismatch_name}_{dataset_name}_{judge_llm_name}_{prompt_name}_{kshots}shots.csv"
        output_file = self.outfiles_dir / output_file_name

        # Construct large language chat model
        llm1_config = load_config(llm_configs, llm1_name)
        chat_llm1 = create_llm(llm_name=llm1_name, model_config=llm1_config)
        llm2_config = load_config(llm_configs, llm2_name)
        chat_llm2 = create_llm(llm_name=llm2_name, model_config=llm2_config)
        judge_llm_config = load_config(llm_configs, judge_llm_name)
        chat_llm_judge = create_llm(llm_name=judge_llm_name, model_config=judge_llm_config)

        # Prepare the unlabelled data to annotate
        df = pd.read_csv(self.data_file, index_col=False)
        df = df[[self.feature_col, self.answer_col]]
        if testing:
            if testing_size is None:
                raise ValueError("testing_size cannot be None in the testing mode. It must be a valid integer.")
            if len(df) < testing_size:
                logger.info(f"Provided testing_size ({testing_size}) is greater than the number of questions ({len(questions)}), "
                    f"we will only use the first {len(df)} questions.")
                testing_size = len(df)

            df = df[:testing_size]
            indices = [idx for idx in indices if idx < testing_size]
            mismatch_indices = [idx for idx in mismatch_indices if idx < testing_size]

        # Create the prefix and suffix from the prompt 
        cot_prefix = get_prompt(prompt_file=cot_prompt_file)
        cot_suffix = "Please "
        if kshots > 0:
            cot_prefix = cot_prefix + f" You're given {kshots} examples for references. " 
            cot_suffix = f"That's all {kshots} examples with succinct answers. However, For the next text, you need to first analyze it step by step and provide reasoning. Then "
        cot_suffix = cot_suffix + "perform the task by analyzing the following text step by step and provide your reasoning. Do not exceed 50 words. Finally, based on the reasoning, choose the correct answer from the options (" + ", ".join(self.options) + ') in a single-choice format with options in \'<\' and \'>\'.\nText: """\n{input}\n"""\nAnswer: Let\'s think step by step. '
        
        judge_prefix = get_prompt(prompt_file=judge_prompt_file)
        judge_suffix = ""
        if kshots > 0:
            judge_prefix = judge_prefix + f" You're given {kshots} examples for references. " 
            judge_suffix = f"That's all {kshots} examples with succinct answers. "
        judge_suffix = judge_suffix + "Please judge which response to the following text is correct and provide reasoning within 150 words. Finally, answer the question by copy the correct reponse's choice from (" + ", ".join(self.options) + ') in \'<\' and \'>\'.\nText: """\n{input}\n"""\nAnswer: '

        call_llm_with_limits = call_llm_wrapper(self.requests_per_period, self.secondf_per_period)

        start_time = time.time()
        # Call LLM to annotate unlabelled instances
        results = []
        processed_ids = []
        try:
            for i in tqdm(mismatch_indices, total=len(mismatch_indices), disable=disable_progress_bar, leave=False, desc="Processing indices"):
                if i not in indices:
                    examples = select_kshots(
                        df, self.feature_col, self.answer_col, kshots, i, indices, embeddings, lambda_param, self.options
                    )
                    example_prompt = PromptTemplate(
                        input_variables=["content", "answer"], template='Text: """\n{content}\n"""\nAnswer: {answer}'
                    )

                    cot_prompt = FewShotPromptTemplate(
                        examples=examples,
                        example_prompt=example_prompt,
                        prefix=cot_prefix,
                        suffix=cot_suffix,
                        input_variables=["input"],
                    )
                    judge_prompt = FewShotPromptTemplate(
                        examples=examples,
                        example_prompt=example_prompt,
                        prefix=judge_prefix,
                        suffix=judge_suffix,
                        input_variables=["input", "response1", "response2"],
                    )
                    row = df.loc[i, self.feature_col]
                    question = {"input": row}

                    # Ask both LLMs
                    chain1 = cot_prompt | chat_llm1 | StrOutputParser()
                    response1 = call_llm_with_limits(chain1, question)
                    chain2 = cot_prompt | chat_llm2 | StrOutputParser()
                    response2 = call_llm_with_limits(chain2, question)

                    # Format comparison prompts and ask the judging LLM
                    judge_question = {"input": row, "response1": response1, "response2": response2}
                    chain_judge = judge_prompt | chat_llm_judge | StrOutputParser()
                    judge_response = call_llm_with_limits(chain_judge, judge_question)
                else:
                    result = df.loc[i, self.answer_col]
                results.append(judge_response)
                processed_ids.append(i)

        except Exception as e:  # Naked execpt, blame the your LLM API
            logger.error(f"Error when calling LLM API to annotate texts: {e}")
        finally:
            # Track and print the computation time
            end_time = time.time()
            elapsed = end_time - start_time
            hours, minutes, secondf = track_computation_time(elapsed)
            logger.info(f"Computation time: {int(hours):02}:{int(minutes):02}:{int(secondf):02}")
            if results:
                results_df = pd.DataFrame()
                results_df["mismatch ids"] = processed_ids
                results_df[self.answer_col] = results
                results_df.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
            else:
                raise Exception("Empty results, please check the log.")
        