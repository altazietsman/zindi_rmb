import pandas as pd


def weight_based_headline_cpi(cpi_weights, cpi_cat_pred):
    """Calculates the headline cpi based on weights

    Arguments:
    ----------
    cpi_cat_pred: pandas dataframe
                  predicted cpi values for each category
    cpi_weigths: pandas dataframe
                 weights per category

    Returns:
    --------
    headline_cpi: float
    """

    cpi_weights = cpi_weights.set_axis(
        [entry.strip() for entry in cpi_weights.index], axis="index"
    )

    head_line = 0
    for entry in cpi_weights.index:
        head_line = head_line + (
            cpi_weights.loc[entry] * cpi_cat_pred.iloc[70][entry]
        )  ##TODO: this is not right. It will depend on the prediction table

    return head_line / 100
