"""
Makes a rho-T plot. Uses the swiftsimio library.
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import swiftsimio as sw
from unyt import mh, cm, Gyr, unyt_array
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
plt.style.use('../mnras.mplstyle')


# Specific to this script needed for data generation
# These are attached as metadata to the data file
parameters = {
    'density_bounds': np.array([10 ** (-9.5), 1e7]),           # in nh/cm^3
    'temperature_bounds': np.array([10 ** (0), 10 ** (9.5)]),  # in K
    'n_bin': 256,
}

# Parameters passed when running the script
# These are also attached as metadata to the data file
parser = argparse.ArgumentParser()
base_dir = f'/cosma8/data/dp004/colibre/Runs'
parser.add_argument('--sim', type=str, required=True, help="Simulation name")
parser.add_argument('--snap-nr', type=int, required=True, help="Snapshot number")
parser.add_argument('--generate-data', action='store_true', help="Whether to generate data")
parser.add_argument('--skip-plotting', action='store_true', help="Whether to skip plotting")
args = parser.parse_args()
data_filename = os.path.basename(__file__.removesuffix('.py')) + '.hdf5'
plot_filename = os.path.basename(__file__.removesuffix('.py')) + '.png'
parameters['sim'] = args.sim
parameters['snap_nr'] = args.snap_nr

if args.generate_data:

    # Generating the data needed for this plot, storing it in the data dict
    snap_filename = f'{base_dir}/{args.sim}/snapshots/colibre_{args.snap_nr:04}/colibre_{args.snap_nr:04}.hdf5'
    snap = sw.load(snap_filename)
    number_density = (snap.gas.densities.to_physical() / mh).to(cm ** -3).value
    temperature = snap.gas.temperatures.to_physical().to("K").value

    density_bins = np.logspace(
        np.log10(parameters['density_bounds'][0]), np.log10(parameters['density_bounds'][1]), parameters['n_bin']
    )
    temperature_bins = np.logspace(
        np.log10(parameters['temperature_bounds'][0]), np.log10(parameters['temperature_bounds'][1]), parameters['n_bin']
    )
    hist, density_edges, temperature_edges = np.histogram2d(
        number_density, 
        temperature,
        bins=[density_bins, temperature_bins],
    )
    data = {
        'hist': hist.T,
        'density_edges': density_edges,
        'temperature_edges': temperature_edges,
    }

    # Saving the data
    with h5py.File(data_filename, 'w') as file:
        for k, v in parameters.items():
            file.attrs[k] = v
        for k, v in data.items():
            file.create_dataset(k, data=v)

else:
    # Loading the data
    with h5py.File(data_filename, 'r') as file:
        for k, v in parameters.items():
            if isinstance(file.attrs[k], np.ndarray):
                assert np.all(file.attrs[k] == v), f'Parameter mismatch {k}'
            else:
                assert file.attrs[k] == v, f'Parameter mismatch {k}'
        data = {k: file[k][:] for k in file.keys()}


if args.skip_plotting:
    print('Not plotting data')
    print('Done!')
    exit()
print('Generating plot')


fig_w, fig_h = plt.figaspect(1)
fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

ax.set_xlabel("Density [$n_H$ cm$^{-3}$]")
ax.set_ylabel("Temperature [K]")
ax.loglog()

vmax = np.max(data['hist'])
mappable = ax.pcolormesh(data['density_edges'], data['temperature_edges'], data['hist'], norm=LogNorm(vmin=1, vmax=vmax))
ax.text(
    0.025,
    0.975,
    args.sim,
    ha="left",
    va="top",
    transform=ax.transAxes,
    fontsize=5,
    in_layout=False,
)
ax.set_xlim(*parameters['density_bounds'])
ax.set_ylim(*parameters['temperature_bounds'])

fig.colorbar(mappable, ax=ax, label="Number of particles")
fig.savefig(plot_filename)

print('Done!')

