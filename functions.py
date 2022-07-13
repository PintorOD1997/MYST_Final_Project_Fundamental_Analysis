
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
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
#pip install pingouin
import pingouin as pg
import seaborn as sns
import pylab
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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

def backtest(escenarios : pd.DataFrame,metricas : pd.DataFrame,C : int = None, V : int = None):
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
            vol.append(C)
        else: 
            vol.append(V)
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

def performance(backtest_df : pd.DataFrame = None):
    """
    Evaluación de desempeño de la estrategia
    
    Regresa un dataframe con métricas de desempeño de la estrategia implementada.
    Recibe un dataframe en el que los resultados de la inversión sean dados.
    Medidas de desempeño:
    Ratio de éxito : de todas las veces que se implementó la estrategia, cuántos fueron exitosos
    Retorno de capital : basado en el capital con el que se inicia, en cuánto crecio el mismo
    Promedio ganancias : promedio de los rendimientos obtenidos
    Desvest ganancias : Volatilidad de los rendimientos obtenidos
    
    Returns
    
    per_df : Dataframe con el performance del dataframe evaluado.
    
    """
    succ_ratio = np.array([backtest_df.Resultado == "ganada"]).sum()/len(backtest_df)*100
    ov = (backtest_df["Capital Acumulado"][0] - backtest_df["Capital"][0])
    cv = backtest_df["Capital Acumulado"][-1]
    roe = (cv-ov)/ov*100
    pct = np.array(backtest_df["Capital"]/ov)
    ret_mean = pct.mean()
    ret_desv = pct.std()
    per_df = pd.DataFrame(
        {
            "Ratio Éxtio (%)" : succ_ratio,
            "Retorno de Capital (%)" : roe,
            "Promedio Ganancias (%)" : ret_mean*100,
            "Volatilidad Ganancias (%)" : ret_desv*100
        },
        index = range(1)
    )
    return per_df

def segmentar(df: pd.DataFrame, p : float = 0.8):
    """
    Segmentador de DataFrames
    Esta función segmenta dataframes en el porcentaje provisto
    Default: 80% - 20% (Test - Validación)
    
    RETURNS
    dfTest: Dataframe 80%
    dfVal : Dataframe 20%
    
    """
    dfTest = df.iloc[0:int(len(df)*p)]
    dfVal = df.iloc[int(len(df)*p)+1:-1]
    return dfTest,dfVal

# VISUALIZACIÓN DATOS ORIGINALES:
def visualizacion(data_PIB):
    # x axis values
    x = data_PIB["FECHA"]
    # corresponding y axis values
    y = data_PIB["ACTUAL"]  
    # plotting the points 
    plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=5)
    # naming the x axis
    plt.xlabel("Periodo")
    # naming the y axis
    plt.ylabel('PIB actual eurozona') 
    # giving a title to my graph
    plt.title('PIB EUROZONA') 
    # function to show the plot
    plt.show()

# AGREGAR COLUMNA PARA HACER PROMEDIO MOVIL
dt.data_PIB.insert(2, "Promedio móvil", "") 
# rellenar columna:
dt.data_PIB['Promedio móvil']=dt.data_PIB['ACTUAL'].rolling(window=3).mean().shift(-1)

# VISUALIZACIÓN CON PROMEDIO MÓVIL: 
def visualizacion2(data_PIB):
    # x axis values
    x2 = data_PIB["FECHA"]
    # corresponding y axis values
    y2 = data_PIB["Promedio móvil"]  
    # plotting the points 
    plt.plot(x2, y2, color='green', linestyle='dashed', linewidth = 3,
             marker='o', markerfacecolor='blue', markersize=5)
    # naming the x axis
    plt.xlabel("Periodo")
    # naming the y axis
    plt.ylabel('Promedio móvil') 
    # giving a title to my graph
    plt.title('DATOS CON PROMEDIO MÓVIL 3 PERIODOS') 
    # function to show the plot
    plt.show()

# COMPARACIÓN:
# datos con y sin promedio móvil
def visualizacion3(data_PIB):
    X = data_PIB["FECHA"]  
    # Assign variables to the y axis part of the curve
    y = data_PIB["ACTUAL"]  
    z = data_PIB["Promedio móvil"]   
    # Plotting both the curves simultaneously
    plt.plot(X, y, color='b',lw=3, label='datos sin promedio móvil')
    plt.plot(X, z, color='g',lw=3, label='datos con promedio móvil') 
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Periodo")
    plt.ylabel("PIB actual Eurozona")
    plt.title("Comparación con y sin promedio móvil") 
    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend() 
    # To load the display window
    plt.show()

# Revisar correlación 
data2 = dt.data_PIB[["FECHA","Promedio móvil"]]
data2=data2.drop([0,26])
#coeficiente de correlación de Pearson
correlacion =data2.corr()
#correlación spearman
correlacion_spearman=stats.spearmanr(data2["FECHA"],data2["Promedio móvil"])

# agregamos una columa extra para hacer la correlacion parcial
# usando un desplazamiento para tener la informacion de un periodo pasado en c/u
data2.insert(2, "Shift", "") 
data2['Shift']=data2['Promedio móvil'].shift(1)
data22 = data2.drop([1])
correlacion_parcial=pg.partial_corr(data=data22, x='FECHA', y='Promedio móvil', covar='Shift')
correlacion_parcial2=data22.pcorr().round(3)

#
# PRUEBA PARA SABER SI TIENE Homocedasticidad o Heterocedasticidad
# FLIGNER TEST
# ==============================================================================
data3=dt.data_PIB.drop([0,26])
fligner_test = stats.fligner(data3["FECHA"], data3["Promedio móvil"], center='median')

# PRUEBA PARA SABER SI TIENE Homocedasticidad o Heterocedasticidad
# Levene test
# ==============================================================================
levene_test = stats.levene(data3["FECHA"], data3["Promedio móvil"], center='median')

# PRUEBA PARA SABER SI TIENE Homocedasticidad o Heterocedasticidad
# Bartlett test
# ==============================================================================
bartlett_test = stats.bartlett(data3["FECHA"], data3["Promedio móvil"])

# prueba grafica de normalidad
def visualizacion4(data3):    
    stats.probplot(data3["Promedio móvil"],dist="norm",plot=pylab)
    pylab.show()

#Comprobamos normalidad con la prueba de Shapiro- Wilk
shapiro_test=shapiro(data3["Promedio móvil"])

# MODELO POLINOMIAL GRADO 
def polinomio(data3):
    x_model = np.array(data3["FECHA"]).reshape((-1, 1))
    y_model = np.array(data3["Promedio móvil"])
    x_ = PolynomialFeatures(degree=9, include_bias=False).fit_transform(x_model)
    model = LinearRegression().fit(x_, y_model)

    r_sq = model.score(x_, y_model)
    print(f"coefficient of determination: {r_sq}")
    print(f"intercept: {model.intercept_}")
    print(f"coefficients: {model.coef_}")



