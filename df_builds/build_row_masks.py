import pandas as pd
import numpy as np

def build_row_masks(nrows: int = 1024,
                    nunique: int = 2) -> pd.DataFrame:
    """ Builds a dataframe with nrows rows and two columns, 'id' and 'val'.
        The 'id' column contains unique identifiers. The 'val' column contains
        nunique distinct values, which are randomly assigned to each row.
    """
    df = pd.DataFrame({
        'id': range(nrows),
        'val': np.random.randint(0, nunique, size=nrows)
    })
    return df