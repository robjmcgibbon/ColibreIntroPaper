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
    'pressure_bounds': np.array([10 ** (-8.0), 10 ** 8.0]),    # in K/cm^1
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
parameters['sim'] = args.sim
parameters['snap_nr'] = args.snap_nr

# Which phase plots to make, and the parameters required when generating the data
plot_names = {
    'density_temperature': [
        'density_bounds', 
        'temperature_bounds',
        'n_bin',
    ],
}
# Check required parameters are valid
for plot_name, required_params in plot_names.items():
    for param in required_params:
        assert param in parameters, f'{plot_name}:{param} not in parameters'

snap_filename = f'{base_dir}/{args.sim}/snapshots/colibre_{args.snap_nr:04}/colibre_{args.snap_nr:04}.hdf5'
snap = sw.load(snap_filename)

# Hold datasets in case we need them for multiple plots
datasets = {}
def load_dataset(dataset_name):
    if dataset_name in datasets:
        pass
    elif dataset_name == 'density':
        datasets[dataset_name] = (snap.gas.densities.to_physical() / mh).to(cm ** -3).value
    elif dataset_name == 'temperature':
        datasets[dataset_name] = snap.gas.temperatures.to_physical().to("K").value
    else:
        raise NotImplementedError
    return datasets[dataset_name]

# Generate/load the data for the plots
plot_data = {}
if args.generate_data:

    # Generating the data needed for this plot, storing it in the plot_data dict and an hdf5 file
    snap_filename = f'{base_dir}/{args.sim}/snapshots/colibre_{args.snap_nr:04}/colibre_{args.snap_nr:04}.hdf5'
    snap = sw.load(snap_filename)

    for plot_name, required_params in plot_names.items():

        if plot_name == 'density_temperature':
            density = load_dataset('density')
            temperature = load_dataset('temperature')

            density_bins = np.logspace(
                np.log10(parameters['density_bounds'][0]), np.log10(parameters['density_bounds'][1]), parameters['n_bin']
            )
            temperature_bins = np.logspace(
                np.log10(parameters['temperature_bounds'][0]), np.log10(parameters['temperature_bounds'][1]), parameters['n_bin']
            )
            hist, density_edges, temperature_edges = np.histogram2d(
                density, 
                temperature,
                bins=[density_bins, temperature_bins],
            )
            plot_data['density_temperature'] = {
                'hist': hist.T,
                'density_edges': density_edges,
                'temperature_edges': temperature_edges,
            }

        # Saving the data
        with h5py.File(data_filename, 'a') as file:
            if plot_name in file:
                del file[plot_name]
            group = file.create_group(plot_name)
            for k, v in parameters.items():
                if not k in required_params:
                    continue
                group.attrs[k] = v
            for k, v in plot_data[plot_name].items():
                group.create_dataset(k, data=v)

else:
    # Loading the data into the plot_data dict
    for plot_name, required_params in plot_names.items():
        with h5py.File(data_filename, 'r') as file:
            group = file[plot_name]
            for k, v in parameters.items():
                if not k in required_params:
                    continue
                if isinstance(group.attrs[k], np.ndarray):
                    assert np.all(group.attrs[k] == v), f'Parameter mismatch for {plot_name}: {k}'
                else:
                    assert group.attrs[k] == v, f'Parameter mismatch for {plot_name}: {k}'
            data = {k: group[k][:] for k in group.keys()}
        plot_data[plot_name] = data


if args.skip_plotting:
    print('Not plotting data')
    print('Done!')
    exit()
print('Generating plot')


for plot_name in plot_names:
    data = plot_data[plot_name]
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
    fig.savefig(plot_name+'.png')

print('Done!')

