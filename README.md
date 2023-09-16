# ENV
Python version 3.9.16 was used. All requirements can be found in the requirements.txt file. Ensure that all libraries are installed (locally or within a conda env.)

It is important set your PYTHONPATH to this dir for example:

export PYTHONPATH=${PYTHONPATH}:<path to directory>

# Data
All data used can be found in the Data dir. Only historic CPI was used

# Code
To create predictions for a new month, update the month variable in the config. Note that entire month needs to be written and not just abbreviations for example March not Mar and November not Nov.

If the config is updated you can either run the predict script in your terminal (python predict.py) or you can call the predict function in a notebook (see example notebook attached). Before running the predict function ensure that all libraries are installed or conda env is activate (if applicable).

The predict function will load the historic cpi data, make predictions and save the submission file to the submissions folder.

# Submission File
The submission file used can be found in the submissions dir

# Update on submission during RMB competition
Please note that only the config will be updated throughout the competition as well as the example notebook that uses all the functions to make predictions (this is only for demonstration purposes).

The latest cpi file from statsa will also be updated every month to contain the latest data.

# Models
All model classes tested can be found in the models directory. Note that not all models were used for the final submission, however they are still supplied for transparency. Models tested includes Holtwinters and Arima as well as Prophet and Varima models with fuel data, currecy data, south african credit information and sabor metric.

# Repo structure
- zindi rmd
    - config
        - core.py
    - data
        - statssa_cpi.xlsx
    - models
        - AutoArima.py
        - HoltWinters.py
        - Prophet.py
        - Varima.py
    - submissions
    - utils
        - data.py
        - stats.py
    - config.yml
    - predict.py
    - example_notebook.ipynb
    - README.md
    - requirements.text
    