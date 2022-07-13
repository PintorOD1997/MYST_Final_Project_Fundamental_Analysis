
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""


import pandas as pd 

fileA = "files/PIB_Eur_Historico.csv"
fileB = "files/EURUSD_Data.csv"


def toDF(file: str = None) -> True:
    """
    Open as pandas DataFrame function
    This function opens a csv file using its provided filename as a pandas DataFrame

    Parameters
    ----------
    file (str) : Filename

    Returns
    -------
    df : csv file as DataFrame
    """

    df = pd.read_csv(file)
    df.set_index(pd.to_datetime(df["timestamp"]),inplace=True)
    df.drop(columns="timestamp",inplace=True)
    
    return df
dfA = toDF(fileA)
dfB = toDF(fileB)