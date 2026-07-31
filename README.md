# ChatGPT Poverty Ranking

This repository contains reference implemenation for the paper: 

[Sarmadi, Hamid, Ola Hall, Thorsteinn Rögnvaldsson, and Mattias Ohlsson. **“Leveraging ChatGPT’s Multimodal Vision Capabilities to Rank Satellite Images by Poverty Level: Advancing Tools for Social Science Research.”** arXiv:2501.14546. Preprint, arXiv, July 28, 2026.](https://doi.org/10.48550/arXiv.2501.14546)

The code is for comparing satellite image locations by wealth indicators using pairwise ChatGPT/OpenAI judgements, survey wealth labels, model outputs, and ranking analysis.

The main analysis is in `analysis.ipynb`. Supporting functions are in `utils.py`, and `chatGPT_communicator.py` contains code for generating pairwise image comparisons through the OpenAI API and Google Cloud Storage signed image URLs.

## Repository Contents

- `analysis.ipynb`: notebook for loading the unified data, computing rankings, comparing models, and generating figures.
- `utils.py`: analysis utilities for ranking, quantile labels, accuracy summaries, and plotting.
- `chatGPT_communicator.py`: script for pairwise image comparison using OpenAI vision models.
- `data/`: unified analysis table and pairwise comparison data used by the notebook.
- `output_figures/`: generated figures.
- `environment.yml`: conda environment definition.
- `requirements.txt`: pip dependency list.
- `.env.example`: template for required environment variables. After updating should be renamed to `.env`

## Python Dependencies Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate chatgpt-poverty
```

If the environment already exists:

```bash
conda env update -f environment.yml
```

Using pip:

```bash
python -m venv .chatgpt-poverty
source .chatgpt-poverty/bin/activate
python -m pip install -r requirements.txt
```

## Configuration

Recommended variables if you run or adapt `chatGPT_communicator.py`:

```bash
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-service-account.json
GCS_BUCKET_NAME=your_gcs_bucket_name
INPUT_CSV_PATH=data/data.csv
OUTPUT_CSV_PATH=data/comparisons.csv
```

These parameter settings exist in `.env.example` with placeholder values. Update the file before renaming it to `.env` and running the communicator.

## Usage

If you want to see the same results as in the paper, you do not need to do anything else than just running the notebook:

```bash
jupyter notebook analysis.ipynb
```

To run the pairwise comparison script:

```bash
python chatGPT_communicator.py
```

The comparison (communicator) script loads variables from `.env` when present. It expects `INPUT_CSV_PATH` to point to a CSV with a `file_name` column. It writes pairwise comparison results to the configured output CSV. The current notebook reads `data/data.csv` and `data/comparisons.csv`.

## Data

The `data/` folder contains the analysis data table and pairwise comparison results used by the notebook.

## Citation

If you use code or data from this repository, please cite our paper:

```bibtex
@misc{sarmadi2026leveragingchatgptsmultimodalvision,
      title={Leveraging ChatGPT's Multimodal Vision Capabilities to Rank Satellite Images by Poverty Level: Advancing Tools for Social Science Research}, 
      author={Hamid Sarmadi and Ola Hall and Thorsteinn Rögnvaldsson and Mattias Ohlsson},
      year={2026},
      eprint={2501.14546},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.14546}, 
}
```

## Licenses

- The source code in this repository is distributed under the BDS-3-Clause license. See the `LICENSE` file for more details.
- The csv data files within the `data` subfolder in this repository are distributed under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license. See the `DATA-LICENSE` file for more details.
