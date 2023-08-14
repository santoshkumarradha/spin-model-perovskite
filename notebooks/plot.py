import crystal_toolkit.components as ctc
import plotly.graph_objects as go
from crystal_toolkit.settings import SETTINGS
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoClip
from pymatgen.core import Structure
from chgnet.model.model import CHGNet

chgnet = CHGNet.load()




app = Dash(prevent_initial_callbacks=True, assets_folder=SETTINGS.ASSETS_PATH)
temperature_range = [int(t) for t in np.linspace(300, 800, 15)]
trajectory_files = [f"../data/md_files/md_{int(temp)}K.traj" for temp in temperature_range]
trajectory = [read(traj_file, index=-1) for traj_file in trajectory_files]
df_traj = pd.DataFrame(
    {
        "Energy": [
            chgnet.predict_structure(AseAtomsAdaptor.get_structure(atoms))["e"]
            for atoms in trajectory
        ],
        "Temperature": temperature_range,
        # Add other columns as needed, e.g., Forces
    }
)
structure = AseAtomsAdaptor.get_structure(trajectory[0])

step_size = max(1, len(trajectory) // 20)  # ensure slider has max 20 steps
slider = dcc.Slider(id="slider", min=0, max=len(trajectory) - 1, step=step_size, updatemode="drag")
e_col = "Energy"
force_col = "Forces"


def plot_energy_and_forces(
    df: pd.DataFrame, step: int, e_col: str, force_col: str, title: str
) -> go.Figure:
    """Plot energy and forces as a function of relaxation step."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Temperature"], y=df[e_col], mode="lines", name="Energy"))
    line_color = fig.data[0].line.color

    # If you have force data, uncomment the following lines
    # fig.add_trace(
    #     go.Scatter(x=df.index, y=df[force_col], mode="lines", name="Forces", yaxis="y2")
    # )

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis=dict(title="Relaxation Step"),
        yaxis=dict(title=e_col),
        # Uncomment if you have force data
        # yaxis2=dict(title=force_col, overlaying="y", side="right"),
        legend=dict(yanchor="top", y=1, xanchor="right", x=1),
    )

    fig.add_vline(x=step, line=dict(dash="dash", width=1))

    # If you have a specific DFT energy value, uncomment the following lines
    # anno = dict(text="DFT final energy", yanchor="top")
    # fig.add_hline(
    #     y=dft_energy,
    #     line=dict(dash="dot", width=1, color=line_color),
    #     annotation=anno,
    # )

    return fig


def make_title(spg_symbol: str, spg_num: int) -> str:
    """Return a title for the figure."""
    return ""


title = make_title(*structure.get_space_group_info())

graph = dcc.Graph(
    id="fig",
    figure=plot_energy_and_forces(df_traj, 0, e_col, force_col, title),
    style={"maxWidth": "50%"},
)

struct_comp = ctc.StructureMoleculeComponent(id="structure", struct_or_mol=structure)

app.layout = html.Div(
    [
        html.H1("Structure Relaxation Trajectory", style=dict(margin="1em", fontSize="2em")),
        html.P("Drag slider to see structure at different relaxation steps."),
        slider,
        html.Div([struct_comp.layout(), graph], style=dict(display="flex", gap="2em")),
    ],
    style=dict(margin="auto", textAlign="center", maxWidth="1200px", padding="2em"),
)

ctc.register_crystal_toolkit(app=app, layout=app.layout)


@app.callback(Output(struct_comp.id(), "data"), Output(graph, "figure"), Input(slider, "value"))
def update_structure(step):
    lattice = trajectory[step].get_cell()
    coords = trajectory[step].get_positions()
    structure.lattice = lattice
    for site, coord in zip(structure, coords):
        site.coords = coord

    title = make_title(*structure.get_space_group_info())
    fig = plot_energy_and_forces(df_traj, step, e_col, force_col, title)

    return structure, fig


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
# app.run(mode="inline", height=800, use_reloader=True)
