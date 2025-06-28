import marimo as mo

def component_plot_ui(n_components: int, top_correlated_components: list[tuple[int, int]] = None):
    ui = {
            "x": mo.ui.number(start=0, stop=n_components, value=0, label="X Component"),
            "y": mo.ui.number(start=0, stop=n_components, value=1, label="Y Component"),
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
    if top_correlated_components:
        ui["top_correlated"] = top_correlations_dropdown(
            top_correlated_components,
            ui["x"],
            ui["y"]
        )
    return mo.ui.dictionary(ui)

def plot_components_with_ui(analyzer, ui):
    corr_ui = ui.get("top_correlated")
    corr_ui = [corr_ui] if corr_ui else []
    return mo.vstack(
        [
            mo.ui.altair_chart(
                analyzer.plot_components(
                    ui["x"].value, ui["y"].value, *ui["plot_type"].value
                )
            ),
            mo.hstack(
                [
                    mo.vstack([ui["x"], ui["y"]], align="stretch"),
                    mo.vstack([ui["plot_type"], *corr_ui], align="stretch"),
                ]
            ),
        ],
        align="start",
    )

def top_correlations_dropdown(choices, ui_x, ui_y):
    def on_change(correlation):
        print(correlation)
        print(ui_x)
        ui_x.value = correlation[0]
        ui_y.value = correlation[1]
    return mo.ui.dropdown(
        choices,
        searchable=True,
        label="Top Correlated Components",
        value=choices[0] if choices else None,
        on_change=on_change
    )