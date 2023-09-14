from config.core import config
from pathlib import Path
import pandas as pd
from utils.data import (
    get_montly_cpi,
    load_and_transform_cpi_weights,
    load_models,
    cat_to_id,
    month_dict,
)
from utils.stats import weight_based_headline_cpi


path = str(Path().cwd().resolve())


def predict(month=config["month"], train_range=24):
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
        path + f"/data/statssa_cpi.xlsx",
        dtype="object",
    )
    cpi_weights = load_and_transform_cpi_weights()

    monthly_cpi = get_montly_cpi(raw_cpi=raw_cpi)
    monthly_cpi = monthly_cpi.drop(index=monthly_cpi.index[:train_range]).reset_index(drop=True)
    predictions = {}


    for cat, selected_model in config["model_selection"].items():
        cat_id = cat_to_id[str(cat).strip()]
        train_df = monthly_cpi[monthly_cpi["date"] < f"2023-{month_number}"].sort_values(
            by=["date"]
        )
        model = load_models(model_name=str(selected_model))
        model.fit(train_df[["date", cat_id]])
        pred = model.predict(forecast=1)
        predictions[str(cat).strip()] = pred[0]

    if predictions.get("headline CPI") == None:
        predictions["headline CPI"] = weight_based_headline_cpi(
            cpi_weights=cpi_weights, cpi_cat_pred=predictions
        )

    pred_map = {
        "headline CPI": f"{month}_headline CPI",
        "Alcoholic beverages and tobacco": f"{month}_alcoholic beverages and tobacco",
        "Clothing and footwear": f"{month}_clothing and footwear",
        "Communication": f"{month}_communication",
        "Education": f"{month}_education",
        "Food and non-alcoholic beverages": f"{month}_food and non-alcoholic beverages",
        "Health": f"{month}_health",
        "Household contents and services": f"{month}_household contents and services",
        "Housing and utilities": f"{month}_housing and utilities",
        "Miscellaneous goods and services": f"{month}_miscellaneous goods and services",
        "Recreation and culture": f"{month}_recreation and culture",
        "Restaurants and hotels": f"{month}_restaurants and hotels",
        "Transport": f"{month}_transport",
    }

    df_pred = pd.DataFrame.from_dict(predictions, orient="index").reset_index()

    df_pred.columns = ["ID", "Value"]

    df_pred = df_pred.replace(pred_map)

    df_pred.to_csv(path + f"/submissions/cpi_{month}.csv", index=False)

    return predictions


if __name__ == "__main__":
    predict(month=config["month"])
