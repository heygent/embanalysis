import altair as alt
import pandas as pd


def simple_plot(data: pd.DataFrame, y: str):
    data = data.assign(_nums=range(len(data)))
    return (
        alt.Chart(data)
        .mark_line()
        .encode(x=alt.X("_nums", type="quantitative"), y=alt.Y(y, type="quantitative"))
    )
