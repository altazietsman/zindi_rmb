from config.core import config
from models import HoltWintersWrapper, ProphetWrapper, VarimaWrapper, AutoArimaWrapper
from config.core import month_dict
from pathlib import Path
import pandas as pd
from utils.data import transfrom_cpi


path = str(Path().cwd().resolve())


def predict(month=config["month"]):
    """Makes prediction and saves submission

    steps:
    1. load data
    2. transforms data to monthly cpi
    3. loads model
    4. makes prediction per category
    5. saves submission

    """
    print(path)
    month_number = month_dict[month]
    raw_cpi = pd.read_excel(
        path
        + f"/data/EXCEL - CPI(5 and 8 digit) from Jan 2017 (20230{month_number}).xlsx",
        dtype="object",
    )
    monthly_cpi = transfrom_cpi(raw_cpi=raw_cpi)
