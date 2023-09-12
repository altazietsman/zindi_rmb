from config.core import config
import pandas as pd
from pathlib import Path
from pydoc import locate


id_to_cat = {
    "01": "Food and non-alcoholic beverages",
    "02": "Alcoholic beverages and tobacco",
    "03": "Clothing and footwear",
    "04": "Housing and utilities",
    "05": "Household contents and services",
    "06": "Health",
    "07": "Transport",
    "08": "Communication",
    "09": "Recreation and culture",
    "10": "Education",
    "11": "Restaurants and hotels",
    "12": "Miscellaneous goods and services",
    "headline": "headline CPI",
}

cat_to_id = dict((v, k) for k, v in id_to_cat.items())

month_dict = {
    "January": "01",
    "February": "02",
    "March": "03",
    "April": "04",
    "May": "05",
    "June": "06",
    "July": "07",
    "August": "08",
    "September": "09",
    "October": "10",
    "November": "11",
    "December": "12",
}


def retrieve_cpi_data(month=config["month"]):
    """Function to retrive cpi data and save to data folder (we will submit the raw data every month as well

    Will take in argment to retrive the month specified in the config"""
    pass


def get_montly_cpi(raw_cpi):
    """Function that takes the raw cpi data for each product from statssa and calculates the cpi value per category

    Arguments:
    ----------
    raw_cpi: pandas dataframe
            dataframe containing raw data from statsa

    Return:
    -------
    transposed_cpi_df: pandas dataframe
            dataframe containing the monthly cpi per category
    """

    # 1. remove unecessary columns and rename
    list_cols_to_drop = ["H01", "H02", "H05", "H06", "H07"]
    cat_cpi_df = raw_cpi.copy().drop(list_cols_to_drop, axis=1).copy()

    cat_cpi_df.rename(
        columns={
            "H03": "category_codes",
            "H04": "category_descr",
            "Weight (All urban)": "weights_urban",
        },
        inplace=True,
    )

    # 2. replace .. with zeros
    cat_cpi_df.replace("..", 0, inplace=True)

    # 3. combine maize meal categories
    cat_cpi_df.iloc[17:19] = (
        cat_cpi_df.iloc[17:19].copy().apply(pd.to_numeric, errors="coerce")
    )
    divided_row = (cat_cpi_df.iloc[17].copy() + cat_cpi_df.iloc[18].copy()) / 2
    cat_cpi_df.iloc[15] = [
        divided_row[i] if value == 0 else value
        for i, value in enumerate(cat_cpi_df.iloc[15].copy())
    ]
    cat_cpi_df.drop([cat_cpi_df.index[17], cat_cpi_df.index[18]], inplace=True)

    # Convert the 'weights_urban' column to float
    cat_cpi_df["weights_urban"] = cat_cpi_df["weights_urban"].astype("float")

    # 4. calculate cpi
    # Assign a main category code to each raw data row.
    main_category = []
    for index, row in cat_cpi_df.iterrows():
        if (len(row["category_codes"]) == 8) & (
            row["category_codes"][:2] in ["01", "02"]
        ):
            main_category.append(row["category_codes"][:2])
        elif (
            len(row["category_codes"]) == 5
        ):  # & (row['category_codes'][:2] not in ["04","07"]):
            main_category.append(row["category_codes"][:2])
        else:
            main_category.append("no")

    cat_cpi_df["main_category_code"] = main_category

    # Drop the rows where the main_category_code is "no". That is to prevent double counting.
    # Some categories have a sub category included in the data.
    cat_cpi_df.drop(
        cat_cpi_df[cat_cpi_df["main_category_code"] == "no"].index, inplace=True
    )

    # Sum the weights for each category
    sum_weights = cat_cpi_df.groupby("main_category_code")["weights_urban"].sum()

    # create new cpi dataframe
    cpi_df = pd.DataFrame()

    # For each month create the headline CPI value and the CPI value per category.
    for col in range(3, cat_cpi_df.shape[1] - 1):
        cat_cpi_df = cat_cpi_df.copy()
        column_name = cat_cpi_df.columns[col]
        cat_cpi_df["weighted_index_" + column_name] = (
            cat_cpi_df["weights_urban"] * cat_cpi_df[column_name]
        )

        sum_weighted_index = cat_cpi_df.groupby("main_category_code")[
            "weighted_index_" + column_name
        ].sum()

        # Concatenate the DataFrames horizontally
        concat_df = pd.concat([sum_weights, sum_weighted_index], axis=1)

        # Add a row that sums the values in the columns
        sums_df = pd.DataFrame(
            concat_df.sum().values.reshape(1, -1), columns=concat_df.columns
        )
        sums_df = sums_df.set_index(pd.Index(["headline"]))

        # Concatenate the headline dataframe to the categories
        month_cpi_df = pd.concat([concat_df, sums_df], axis=0)

        # Calculate the CPI value
        month_cpi_df["cpi_" + column_name] = (
            month_cpi_df["weighted_index_" + column_name]
            / month_cpi_df["weights_urban"]
        ).round(1)

        cpi_df = pd.concat(
            [cpi_df, month_cpi_df[["weights_urban", "cpi_" + column_name]]], axis=1
        )

    # Remove duplicate weights columns and reset the index
    cpi_df = cpi_df.loc[:, ~cpi_df.columns.duplicated()]
    cpi_df = cpi_df.reset_index().rename(columns={"index": "category"})

    # Dataframe with just the CPI values:
    cpi_df = cpi_df.drop("weights_urban", axis=1).copy()
    transposed_cpi_df = cpi_df.set_index("category").transpose().reset_index()
    transposed_cpi_df["date"] = transposed_cpi_df["index"].apply(
        lambda x: x.split("M")[-1]
    )
    transposed_cpi_df["date"] = transposed_cpi_df["date"].apply(
        lambda x: x[:4] + "-" + x[-2:]
    )
    # change month to datetime format
    transposed_cpi_df["date"] = pd.to_datetime(transposed_cpi_df["date"]).dt.strftime(
        "%Y-%m"
    )

    return transposed_cpi_df


def load_and_transform_cpi_weights():
    """Function to load cpi weights (provided by zindi).

    Returns:
    --------
    cpi_weights: pandas dataframe
    """

    path = str(Path().cwd().resolve())

    cpi_weights = pd.read_excel(
        path + f"/data/cpi_weights.xlsx",
        dtype="object",
    )

    return cpi_weights.replace({"Headline_CPI": "headline CPI"})


def load_models(model_name: str):
    model_params = model_name.split("_")

    model = model_params[0]

    model_import = locate(f"models.{model}.{model}")

    if model == "HoltWinters":
        return model_import(
            trend=model_params[1],
            seasonal=model_params[2],
            seasonal_periods=int(model_params[3]),
        )

    elif model == "AutoArima":
        return model_import()

    elif model == "Prophet":
        return model_import(
            changepoint_range=int(model_params[1]),
            n_changepoints=int(model_params[2]),
            changepoint_prior_scale=float(model_params[3]),
        )

    elif model == "Varima":
        return model_import()

    # TODO: calculate cpi for headline based on weights
