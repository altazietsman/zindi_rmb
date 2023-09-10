from config.core import config
import pandas as pd


category_dict = {
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


def retrieve_cpi_data(month=config["month"]):
    """Function to retrive cpi data and save to data folder (we will submit the raw data every month as well

    Will take in argment to retrive the month specified in the config"""
    pass


def transfrom_cpi(raw_cpi, cpi_cat_dict: dict = category_dict):
    """Function to get monthly cpi data from the raw data. Data willm be saved to data folder

    The following steps are taking to convert the raw cpi data from stats sa to monthly cpi:
    - Remove unnecessary columns.
    - Change column headers to make them more explanatory.
    - Replace all `..` entries with a `0`.
    - Combine the `Super maize` and `Special maize` categories into a single `Maize meal` category, to correspond with the current use of maize meal.
    - Calculate the CPI values for each month using the weights provided in the file.

    Arguments:
    ----------
    raw_cpi: pandas dataframe
             raw cpi data pulled from stats sa
    cpi_cat_dict: dict
                  maps category names to numbers
    Returns:
    --------
    cpi_df: pandas df
            monthly cpi values
    """

    list_cols_to_drop = ["H01", "H02", "H05", "H06", "H07"]
    cat_cpi_df = raw_cpi.drop(list_cols_to_drop, axis=1).copy()

    cat_cpi_df.rename(
        columns={
            "H03": "category_codes",
            "H04": "category_descr",
            "Weight (All urban)": "weights_urban",
        },
        inplace=True,
    )

    cat_cpi_df.replace("..", 0, inplace=True)

    # Convert the Super maize and Special maize row to numeric data types.
    # The rows will be dropped, so the loss of other information is not a problem.

    cat_cpi_df.iloc[17:19] = cat_cpi_df.iloc[17:19].apply(
        pd.to_numeric, errors="coerce"
    )

    # Divide rows A and B
    divided_row = (cat_cpi_df.iloc[17] + cat_cpi_df.iloc[18]) / 2

    # Replace the value in row C if it is 0 with the divided value
    cat_cpi_df.iloc[15] = [
        divided_row[i] if value == 0 else value
        for i, value in enumerate(cat_cpi_df.iloc[15])
    ]

    # Remove the rows for Super maize and Special maize
    cat_cpi_df.drop([cat_cpi_df.index[17], cat_cpi_df.index[18]], inplace=True)

    cat_cpi_df["weights_urban"] = cat_cpi_df["weights_urban"].astype("float")

    # Make a copy of the input dataframe
    df = cat_cpi_df.copy()

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

    df["main_category_code"] = main_category

    # Drop the rows where the main_category_code is "no". That is to prevent double counting.
    # Some categories have a sub category included in the data.
    df.drop(df[df["main_category_code"] == "no"].index, inplace=True)

    # Sum the weights for each category
    sum_weights = df.groupby("main_category_code")["weights_urban"].sum()

    cpi_df = pd.DataFrame()

    # For each month create the headline CPI value and the CPI value per category.
    for col in range(3, df.shape[1] - 1):
        column_name = df.columns[col]
        df["weighted_index_" + column_name] = df["weights_urban"] * df[column_name]

        sum_weighted_index = df.groupby("main_category_code")[
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

    return cpi_df


def load_cpi_weights():
    """Function to load cpi weights (provided by zindi)."""
    pass
