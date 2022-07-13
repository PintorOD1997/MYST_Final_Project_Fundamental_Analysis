
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import data as dt
import numpy as np
import pandas as pd

def pares(df: pd.DataFrame = None) -> True:
    """
    Pares

    Parameters
    ----------
    df (DataFrame) : DataFrame que contiene los timestamps a utilizar

    Returns
    -------
    startEndpaar : Lista de pares de timestamps para slicear 
    """
    startEndpaar = []
    for i in df.index:
        startEndpaar.append([i-pd.DateOffset(minutes=30),i+pd.DateOffset(minutes=30)])
    return startEndpaar

def trading_history(gdp_df: pd.DataFrame, df: pd.DataFrame = None, pares: list = None) -> True:
    """
    Historia de tradeo
    Información de los trades elaborados el día del anuncio macroeconómico

    Parameters
    ----------
    df (DataFrame) : Dataframe del trading

    Returns
    -------
    timestamp_dic : diccionario que contiene la historia de trading del día del anuncio
    """

    timestamp_dic = {}
    temp = list(gdp_df.index)
    for i in range(len(temp)):
        key = str(temp[i])
        value = df.loc[pares[i][0]:pares[i][1]]
        try:
            timestamp_dic[key].append(value)
        except KeyError:
            timestamp_dic[key] = value
    return timestamp_dic

def escenarios_ocurrencia(df : pd.DataFrame):
    
    """
    Escenarios de Ocurrencia
    Caracterización de escenarios en base a lo anunciado en el aspecto macroeconómico
    A : Actual >= Consenso >= Previo
    B : Actual >= Consenso < Previo
    C : Actual < Consenso >= Previo
    D : Actual < Consenso < Previo

    Parameters
    ----------
    df (DataFrame) : Dataframe de los resultados del anuncio macroeconómico

    Returns
    -------
    out : DataFrame con los escenarios caracterizados
    """
    
    out = df.copy()
    scen = []
    for i in range(len(df)):
        if df["Actual"][i] >= df["Consenso"][i] and df["Consenso"][i]>= df["Previo"][i]:
            scen.append("A")
        elif df["Actual"][i] >= df["Consenso"][i] and df["Consenso"][i]< df["Previo"][i]:
            scen.append("B")
        elif df["Actual"][i] < df["Consenso"][i] and df["Consenso"][i]>= df["Previo"][i]:
            scen.append("C")
        elif df["Actual"][i] < df["Consenso"][i] and df["Consenso"][i]< df["Previo"][i]: 
            scen.append("D")
    out["Escenario"] = scen
    return out



def pip_Metrics(df : pd.DataFrame,price : pd.DataFrame):
    """
    Métricas en PIPS
    Pips alcistas, bajistas y volatilidad del día de t radeo en pips (unidad forex)

    Parameters
    ----------
    df (DataFrame) : Dataframe de los resultados del anuncio macroeconómico
    price (DataFrame) : diccionario de dataframes indexado por timestamp
    Returns
    -------
    out : DataFrame con los resultados de las métricas de pips
    """
    
    df = df.copy()
    out = pd.DataFrame()
    out.index = df.index
    out["Escenario"] = df["Escenario"]
    dir = []
    pip_al = []
    pip_baj = []
    vol = []
    for i in price:
        tempDF = price[i]
        # Open = 0, High = 1, Low = 2, Close = 3
        # Dir
        if (tempDF.iloc[-1,3] - tempDF.iloc[0,0]) > 0: # Evaluación de close > open
            dir.append(1)
        else:
            dir.append(-1)
        # Pips Alcistas
        pip_al.append((np.max(tempDF.iloc[29:-1,1])-tempDF.iloc[0,0])*100000)
        # Pips Bajistas
        pip_baj.append((tempDF.iloc[0,0]-np.max(tempDF.iloc[0:-1,2]))*100000)
        # Volatilidad
        vol.append(np.max((tempDF.iloc[0:-1,1])-tempDF.iloc[:,2].min())*100000)
    out["Direction"] = dir
    out["Pips Alcistas"] = pip_al
    out["Pips Bajistas"] = pip_baj
    out["Volatilidad"] = vol
    return out