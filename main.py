
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

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



backtest_df = ft.backtest(escenarios,y)
test,val = ft.segmentar(backtest_df)
perf_test = ft.performance(test)
perf_val = ft.performance(val)
    
  





            



