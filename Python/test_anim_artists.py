import sys
from ocean.simulator.transit_simulator_app import TransitSimulatorApp
import matplotlib.pyplot as plt

app = TransitSimulatorApp.__new__(TransitSimulatorApp)
app.unit_scale = 0.01
app.time_warp = 1.0
app.fig = plt.figure()
app.ax_map = app.fig.add_axes([0.05, 0.05, 0.69, 0.9])
app.FG, app.DIM, app.CARD, app.GREEN, app.CYAN, app.AMBER, app.PURP = ['#FFF']*7

app._w = {}
app._tele_texts = []
app._scat = app.ax_map.scatter([], [])

# Call the UI builder manually
app._build_panel()

artists = app._get_anim_artists()
for a in artists:
    if a.axes is None:
        print(f'BAD ARTIST: {a}')
    else:
        print(f'GOOD: {a}')
