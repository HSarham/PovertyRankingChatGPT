# Data Files

This folder contains the unified analysis data and pairwise comparison results used by `analysis.ipynb`.

## Files

- `data.csv`: cluster-level data table used by the notebook and as the default image-name input for `chatGPT_communicator.py`. It includes a `file_name` column, scaled survey-based wealth indices (`v191` and `hv271`), CNN outputs, random forest outputs, and domain expert rankings.
- `comparisons.csv`: ChatGPT pairwise image comparison results with columns `image1`, `image2`, and `winner`; used to compute the rankings. This file is the output of `chatGPT_communicator.py`.

