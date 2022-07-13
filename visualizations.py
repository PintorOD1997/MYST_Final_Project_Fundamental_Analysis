
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
  
import plotly.io as pio
import plotly.graph_objects as go 

def backtest_evolution_chart(backtest_df : pd.DataFrame = None, title : str = None):
    fig = go.Figure(
            data=[
                go.Scatter(
                    x = backtest_df.index,
                    y = backtest_df["Capital Acumulado"],
                    name = "Evolución Capital"               
                ),
                go.Scatter(
                    x = backtest_df.index,
                    y = np.array([100000]*len(backtest_df)),
                    name = "Capital Inicial"
                ),
                go.Bar(
                    x = backtest_df.index,
                    y = backtest_df["Capital"],
                    name = "Resultado Apuesta",
                    marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                                line=dict(color='rgb(0,0,0)',width=1.5)),
                    text = backtest_df["Operación"]
                )
            ]
        )
    fig.update_layout(
            xaxis_title = "Timestamp",
            yaxis_title = "Value",
            legend_title = "Variable",
            title = title
        )
    fig.update_xaxes(showspikes=True, spikecolor="green", spikesnap="cursor", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="orange", spikethickness=2)
    fig.update_layout(spikedistance=1000, hoverdistance=100)
    return fig

def backtest_strat_result(backtest_df : pd.DataFrame = None,  title : str = None):
    fig = go.Figure(
            data=[
                go.Bar(
                    x = np.arange(len(backtest_df)),
                    y = backtest_df["Capital"],
                    name = "Resultado Apuesta",
                    marker = dict(color = [1 if backtest_df["Resultado"][i] == "ganada" else 2 for i in range(len(backtest_df))],
                                line=dict(color='rgb(0,0,0)',width=1.5)),
                    text = backtest_df["Operación"],
                    hovertext = backtest_df["Resultado"]
                    
                )
            ]
        )
    fig.update_layout(
            xaxis_title = "Operación no.",
            yaxis_title = "Valor",
            title = title
        )
    return fig

                