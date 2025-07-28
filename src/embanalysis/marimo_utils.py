import marimo as mo

from embanalysis.analyzer import EmbeddingsAnalyzer


def plot_type_ui(default_value="Token value gradient"):
    return mo.ui.dropdown(
        {
            "Token value gradient": ("gradient",),
            "Digit length": ("digit_length",),
            "Ones Digit": ("digit", 0),
            "Tens Digit": ("digit", 1),
            "Hundreds Digit": ("digit", 2),
        },
        label="Coloring",
        value=default_value,
    )


def component_plot_ui(n_components: int, default_x=0, default_y=1):
    ui = {
        "x": mo.ui.number(
            start=0, stop=n_components, value=default_x, label="X Component"
        ),
        "y": mo.ui.number(
            start=0, stop=n_components, value=default_y, label="Y Component"
        ),
        "plot_type": plot_type_ui()
    }
    return mo.ui.dictionary(ui)


def plot_components_with_ui(analyzer, ui):
    return mo.vstack(
        [
            mo.ui.altair_chart(
                analyzer.plot.components(
                    ui["x"].value, ui["y"].value, *ui["plot_type"].value
                )
            ),
            ui.hstack(),
        ],
    )

def plot_components_with_type_ui(analyzer, x_component, y_component, type_ui):
    return mo.vstack(
        [
            mo.ui.altair_chart(
                analyzer.plot.components(
                    x_component, y_component, *type_ui.value
                )
            ),
            type_ui
        ],
    )

def plot_3d_components_with_type_ui(analyzer: EmbeddingsAnalyzer, x_component, y_component, z_component, type_ui):
    return mo.vstack(
        [
            mo.ui.plotly(
                analyzer.plot.components_3d(
                    x_component, y_component, z_component, *type_ui.value
                )
            ),
            type_ui
        ],
    )