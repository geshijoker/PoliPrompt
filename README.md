
<p align="center">
  <img src="https://github.com/geshijoker/PoliPrompt/blob/main/poliprompt_logo.png" width="800" />
</p>

---

[![PyPI - Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://pypi.org/project/poliprompt/)
[![Build](https://img.shields.io/github/actions/workflow/status/geshijoker/PoliPrompt/ci.yml?branch=main)](https://github.com/geshijoker/PoliPrompt/actions)
[![docs](https://img.shields.io/badge/docs-Passing-green.svg)](https://poliprompt-tutorial.readthedocs.io/en/latest/)
[![PyPI - PyPi](https://img.shields.io/pypi/v/PoliPrompt)](https://pypi.org/project/poliprompt/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/geshijoker/PoliPrompt/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2409.01466-<COLOR>.svg)](https://arxiv.org/pdf/2409.01466)


# PoliPrompt
PoliPrompt is a high-performance, cost-effective LLM-based text classification framework designed for political science research. It automates prompt optimization and dynamically selects exemplars during few-shot learning inference to improve the accuracy of classification tasks, especially in the context of sentiment analysis, stance detection, and more.

## Getting Started
### A. Set Up Virtual Environment and Install PoliPrompt Package

1. **Use the following command to display your current directory (user home directory):**
   
```bash
pwd
```

2. **Navigate to the Directory (e.g. user home directory) Where You Want to Set Up the Virtual Environment**
   
```bash
cd ~
```

3. **Run the following command to create a virtual environment named poliprompt_env (changeable):**
   
```bash
python3 -m venv poliprompt_env
```

4. **Activate the Virtual Environment**
   
```bash
source poliprompt_env/bin/activate
```

5. **After activating the environment, install the PoliPrompt package:**
```bash
pip install PoliPrompt
```

6. **Download the Poliprompt source code for examples**
   - To run the example code, users need to download the source code for notebooks and examples
   - Visit [project home page](https://github.com/geshijoker/PoliPrompt/tree/main) on github
   - Download the project by `git clone HTTP/SSH` in terminal or Download ZIP (please refer to github basic tutorials)
   - Put it in user specified location on local machine, for example, the user home directory `~`.

## B. Register and Obtain API Keys

To use PoliPrompt effectively, you will need API keys from various large language model platforms. Follow these steps to register and retrieve your keys:

1. **Register on LLM Platforms:**
   - Visit the links below to sign up for API access from popular large language model providers:
     - [OpenAI](https://beta.openai.com/signup/) – Sign up for GPT-based APIs.
     - [Anthropic](https://www.anthropic.com/product) – Register for Claude-based APIs.
     - [Voyage AI](https://voyage.ai) – Obtain access to additional AI tools.

2. **Generate and SAVE API Keys:**
   - Once registered, navigate to the API section of each platform's dashboard.
   - Generate new API keys.
   - Copy the keys for use in your `.env` file. 

3. **Create and Save the `.env` File:**
   - In your terminal, navigate to the root directory of your project (e.g., `~/PoliPrompt`).
   - Run the following command to open a new `.env` file for editing:
   
   ```bash
   vim .env
   ```

   - In the `.env` file, add your API keys in the following format:

   ```bash
   OPENAI_API_KEY=your-openai-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   VOYAGE_API_KEY=your-voyage-api-key
   ```

## C. Run the Example Code

### 1. **Install Jupyter Notebook**
If Jupyter Notebook is not already installed, you can install it using pip:

```bash
pip install notebook
```
2. **To use the PoliPrompt virtual environment in Jupyter, you need to create a new kernel:**

```bash
pip install ipykernel
python -m ipykernel install --user --name=PoliPrompt_env --display-name "PoliPrompt Kernel"
```

3. **Navigate to the folder where the TopicExperiment.ipynb file is located. For example, if the file is in the notebooks folder inside the PoliPrompt directory, run:**

```bash
cd ~/PoliPrompt/notebooks
```

4. **Start Jupyter Notebook by running the following command:**

```bash
jupyter notebook
```

5. **Select the PoliPrompt Kernel**
   -Once the notebook is open:
   -Navigate to the top-right corner of the notebook interface.
   -Click on "Kernel" → "Change Kernel."
   -Select PoliPrompt Kernel from the dropdown list.

6. **After selecting the kernel, you can run the example code (TopicExperiment.ipynb) inside the notebook.**



