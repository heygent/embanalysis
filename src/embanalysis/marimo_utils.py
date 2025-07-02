import marimo as mo


def component_plot_ui(n_components: int, default_x=0, default_y=1):
    ui = {
        "x": mo.ui.number(
            start=0, stop=n_components, value=default_x, label="X Component"
        ),
        "y": mo.ui.number(
            start=0, stop=n_components, value=default_y, label="Y Component"
        ),
        "plot_type": mo.ui.dropdown(
            {
                "Token value gradient": ("gradient",),
                "Digit length": ("digit_length",),
                "Ones Digit": ("digit", 0),
                "Tens Digit": ("digit", 1),
                "Hundreds Digit": ("digit", 2),
            },
            label="Coloring",
            value="Token value gradient",
        ),
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
