from config.core import config
from pathlib import Path
import pandas as pd
from utils.data import get_montly_cpi, load_and_transform_cpi_weights, load_models, cat_to_id, month_dict


path = str(Path().cwd().resolve())


def predict(month=config["month"]):
    """Makes prediction and saves submission

    steps:
    1. load data
    2. transforms data to monthly cpi
    3. loads model and train
    4. makes prediction per category
    5. saves submission

    """
    month_number = str(month_dict[month])
    raw_cpi = pd.read_excel(
        path
        + f"/data/EXCEL - CPI(5 and 8 digit) from Jan 2017 (20230{month_number.strip('0')}).xlsx",
        dtype="object",
    )
    cpi_weights = load_and_transform_cpi_weights()

    monthly_cpi = get_montly_cpi(raw_cpi=raw_cpi)

    predictions = {}

    for cat, selected_model in config['model_selection'].items():
        cat_id = cat_to_id[cat]
        train_df = monthly_cpi[monthly_cpi['date'] < '2023-{month_number}'].sort_values(by=['date'])

        model = load_models(model_name=str(selected_model))
        model.fit(train_df[['date',cat_id]])
        pred = model.predict(forecast=1)
        predictions[str(cat)] = pred[0]

        
    ## TODO: save predictions



