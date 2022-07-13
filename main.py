
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

from regex import I
import data as dt
import numpy as np
import functions as ft
import pandas as pd

"""

Creación de Previo para trading

Evalúo el escenario, luego que?

"""

# -- TEST 1 : 
#Datraframe testing

currency_df = dt.dfB
gdp_df = dt.dfA

# Obtención de timestamps para comparación

pares = ft.pares(gdp_df)
    
# Elaboración de los dataframes de 60 renglones

dic = ft.trading_history(gdp_df,currency_df
                         ,pares)

# Funciones de escenarios de ocurrencia
# Se le alimenta el df del gdp

escenarios = ft.escenarios_ocurrencia(gdp_df)

# Métricas para escenarios

y = ft.pip_Metrics(escenarios,dic)

df_decisiones = pd.DataFrame({
    "Escenario" : ["A","B","C","D"],
    "Operación" : ["Compra","Venta","Compra","Venta"],
    "Stop Loss" :  [50,50,50,50], 
    "Take Profit" : [150,150,150,150],
    "Volumen" : [10000,5000,500,1000]
}) 
    
# Optimización y backtest


def backtest(escenarios,metricas):
    cash = 100000
    df_backtest = pd.DataFrame()
    df_backtest.index = escenarios.index
    df_backtest["Escenario"] = escenarios["Escenario"]
    op = []
    for i in range(len(escenarios)):
        if df_backtest.iloc[i,0] == "A" or df_backtest.iloc[i,0] == "C":
            op.append("Compra")
        else:
            op.append("Venta")
    df_backtest["Operación"] = op
    vol = []
    res = []
    pips = []
    cap = []
    acm = []
    for i in range(len(escenarios)):
        if df_backtest["Operación"][i] == "Compra":
            vol.append(100)
        else: 
            vol.append(50)
        if (metricas["Pips Alcistas"][i]-metricas["Pips Bajistas"][i])>0 and op[i] == "Compra":
            res.append("ganada")
        elif (metricas["Pips Alcistas"][i]-metricas["Pips Bajistas"][i])<0 and op[i] == "Venta":
            res.append("ganada")
        else: 
            res.append("perdida")
        pips.append(
            (metricas["Pips Alcistas"][i]-metricas["Pips Bajistas"][i])
        )
        cap.append(
            pips[i]*vol[i]
        )
        if res[i] == "perdida":
            cap[i] = cap[i]*-1
        cash += cap[i]
        acm.append(cash)
    df_backtest["Volumen"] = vol
    df_backtest["Resultado"] = res
    df_backtest["Pips"] = pips
    df_backtest["Capital"] = cap
    df_backtest["Capital Acumulado"] = acm
    df_backtest["Capital Acumulado"].astype(int)
    return df_backtest



backtest_df = backtest(escenarios,y)
            



