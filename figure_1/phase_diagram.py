"""
Makes a phase diagrams. Uses the swiftsimio library.

We have multiple difference types of 2d histograms.
This script plots them all. It is in two parts. The
first part loads the data and calculates the histograms.
The arrays are saved once they are loaded.
This data, along with any parameters that were used to
generate it, are written to a file. The second part
loads the generated histograms and plots them.
"""

import argparse
import time
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import psutil
import swiftsimio as sw
import unyt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.animation import FuncAnimation
plt.style.use('../mnras.mplstyle')

parameters = {
    # Data generation
    # These are attached as metadata to the data file
    'n_bin': 256,
    'density_bounds': np.array([10 ** (-9.5), 1e7]),            # nh/cm^3
    'temperature_bounds': np.array([10 ** (0), 10 ** (9.5)]),   # K
    'pressure_bounds': np.array([10 ** (-8.0), 10 ** 16.0]),    # K/cm^1
    'min_metal_frac': -8,                                       # dimensionless (log)
    # Plotting only
    'density_xlim': np.array([10 ** (-9.5), 1e7]),              # nh/cm^3
    'temperature_ylim': np.array([10 ** (0), 10 ** (9.5)]),     # K
    'pressure_ylim': np.array([10 ** (-8.0), 10 ** 16.0]),      # K/cm^1
    'metal_frac_cbar_lim': np.array([-6, -1]),                  # dimensionless (log)
    'dust_to_metal_cbar_lim': np.array([1e-2, 7e-1]),           # dimensionless
    'HI_frac_cbar_lim': np.array([1e-2, 1]),                    # dimensionless
}

# Arguments passed when running the script
parser = argparse.ArgumentParser()
base_dir = f'/cosma8/data/dp004/colibre/Runs'
parser.add_argument('--sim', type=str, required=True, help="Simulation name")
parser.add_argument('--snap-nr', type=int, required=True, help="Snapshot number")
parser.add_argument('--generate-data', action='store_true', help="Whether to generate data")
parser.add_argument('--skip-plotting', action='store_true', help="Whether to skip plotting")
args = parser.parse_args()
data_filename = args.sim.replace('/', '_') + f'_s{args.snap_nr}_'
data_filename += os.path.basename(__file__.removesuffix('.py')) + '.hdf5'
parameters['sim'] = args.sim
parameters['snap-nr'] = args.snap_nr

# Which phase plots to make, and the parameters required when generating the data
plot_names = {
    'density_temperature': [
        'n_bin',
        'density_bounds',
        'temperature_bounds',
    ],
    'density_pressure': [
        'n_bin',
        'density_bounds',
        'pressure_bounds',
    ],
    'density_temperature_metal_frac': [
        'n_bin',
        'density_bounds',
        'temperature_bounds',
        'min_metal_frac',
    ],
    'density_temperature_dust_to_metal': [
        'n_bin',
        'density_bounds',
        'temperature_bounds',
    ],
    'density_temperature_HI_frac': [
        'n_bin',
        'density_bounds',
        'temperature_bounds',
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
        datasets[dataset_name] = (snap.gas.densities.to_physical() / unyt.mh).to(unyt.cm ** -3).value
    elif dataset_name == 'temperature':
        datasets[dataset_name] = snap.gas.temperatures.to_physical().to("K").value
    elif dataset_name == 'pressure':
        # Avoid overflows
        pressure = snap.gas.pressures.to_physical().astype('float64')
        datasets[dataset_name] = (pressure / unyt.kb).to(unyt.K * unyt.cm ** -3).value
    elif dataset_name == 'mass':
        datasets[dataset_name] = snap.gas.masses.to_physical().to("Msun").value
    # Dust
    elif dataset_name == 'dust_frac':
        dfracs = np.zeros_like(snap.gas.masses.value)
        for col in snap.gas.dust_mass_fractions.named_columns:
            dfracs += getattr(snap.gas.dust_mass_fractions, col).value
        datasets[dataset_name] = dfracs
    elif dataset_name == 'dust_mass':
        dfracs = load_dataset('dust_frac')
        mass = load_dataset('mass')
        datasets[dataset_name] = dfracs * mass
    # Metals
    elif dataset_name == 'metal_frac':
        datasets[dataset_name] = snap.gas.metal_mass_fractions.value
    elif dataset_name == 'floor_metal_frac':
        min_metal_frac = 10 ** parameters['min_metal_frac']
        metal_frac = load_dataset('metal_frac').copy()
        metal_frac[metal_frac < min_metal_frac] = min_metal_frac
        datasets[dataset_name] = np.log10(metal_frac)
    elif dataset_name == 'metal_mass':
        metal_frac = load_dataset('metal_frac')
        mass = load_dataset('mass')
        datasets[dataset_name] = metal_frac * mass
    # HI
    elif dataset_name == 'hydrogen_frac':
        datasets[dataset_name] = snap.gas.element_mass_fractions.hydrogen.value
    elif dataset_name == 'HI_mass':
        sfrac = snap.gas.species_fractions.HI.value * load_dataset('hydrogen_frac')
        datasets[dataset_name] = sfrac * load_dataset('mass')
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
        print(f'Generating data for {plot_name}')
        t_start = time.time()

        def create_2Dhistogram(
                plot_name,
                dataset_name_x,
                dataset_name_y, 
                dataset_name_weights_1=None,
                dataset_name_weights_2=None,
            ):

            x = load_dataset(dataset_name_x)
            y = load_dataset(dataset_name_y)
            x_bins = np.logspace(
                np.log10(parameters[f'{dataset_name_x}_bounds'][0]), 
                np.log10(parameters[f'{dataset_name_x}_bounds'][1]), 
                parameters['n_bin']
            )
            y_bins = np.logspace(
                np.log10(parameters[f'{dataset_name_y}_bounds'][0]), 
                np.log10(parameters[f'{dataset_name_y}_bounds'][1]), 
                parameters['n_bin']
            )

            if dataset_name_weights_1 is None:
                H_norm, x_edges, y_edges = np.histogram2d(
                    x, 
                    y,
                    bins=[x_bins, y_bins],
                )
            else:
                weights = load_dataset(dataset_name_weights_1)
                H_norm, x_edges, y_edges = np.histogram2d(
                    x, 
                    y,
                    bins=[x_bins, y_bins], 
                    weights=weights,
                )

            if dataset_name_weights_2 is None:
                # We just want histogram of counts
                hist = H_norm.T
            else:
                weights = load_dataset(dataset_name_weights_2)
                H, _, _ = np.histogram2d(
                    x, 
                    y,
                    bins=[x_bins, y_bins], 
                    weights=weights,
                )
                # Set bins with no values to -100 to avoid division by 0
                invalid = H_norm == 0.0
                H[invalid] = -100
                H_norm[invalid] = 1.0
                hist = (H / H_norm).T

            plot_data[plot_name] = {
                'hist': hist,
                f'{dataset_name_x}_edges': x_edges,
                f'{dataset_name_y}_edges': y_edges,
            }

        if plot_name == 'density_temperature':
            create_2Dhistogram(plot_name, 'density', 'temperature')
        elif plot_name == 'density_pressure':
            create_2Dhistogram(plot_name, 'density', 'pressure')
        elif plot_name == 'density_temperature_metal_frac':
            create_2Dhistogram(
                plot_name, 
                'density', 
                'temperature', 
                dataset_name_weights_2='floor_metal_frac',
            )
        elif plot_name == 'density_temperature_dust_to_metal':
            create_2Dhistogram(
                plot_name, 
                'density', 
                'temperature', 
                dataset_name_weights_1='metal_mass',
                dataset_name_weights_2='dust_mass',
            )
        elif plot_name == 'density_temperature_HI_frac':
            create_2Dhistogram(
                plot_name, 
                'density', 
                'temperature', 
                dataset_name_weights_1='mass',
                dataset_name_weights_2='HI_mass',
            )
        else:
            raise NotImplementedError

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

        print(f'Took {time.time() - t_start:.3g}s')

    GB = 1024 ** 3
    process = psutil.Process()
    print(f'Memory usage: {process.memory_info().rss / GB:.4g} GB')

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
print('Generating plots')


for plot_name in plot_names:
    fig_w, fig_h = plt.figaspect(1)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    data = plot_data[plot_name]
    ax.loglog()

    def plot_2Dhistogram(dataset_name_x, dataset_name_y, norm=None):
        if norm is None:
            norm = LogNorm(vmin=1, vmax=np.max(data['hist']))
        mappable = ax.pcolormesh(
            data[f'{dataset_name_x}_edges'], 
            data[f'{dataset_name_y}_edges'], 
            np.ma.array(data['hist'], mask=data['hist']==-100), 
            norm=norm
        )
        ax.set_xlim(*parameters[f'{dataset_name_x}_xlim'])
        ax.set_ylim(*parameters[f'{dataset_name_y}_ylim'])
        return mappable

    # Most plot are density temperture, so set as defaults
    ax.set_xlabel("Density [$n_H$ cm$^{-3}$]")
    ax.set_ylabel("Temperature [K]")

    if plot_name == 'density_temperature':
        mappable = plot_2Dhistogram('density', 'temperature')
        cbar_label = "Number of particles"
    elif plot_name == 'density_pressure':
        mappable = plot_2Dhistogram('density', 'pressure')
        ax.set_ylabel("Pressure $P / k_B$ [K cm$^{-3}$]")
        cbar_label = "Number of particles"
    elif plot_name == 'density_temperature_metal_frac':
        norm = Normalize(
            vmin=parameters['metal_frac_cbar_lim'][0], 
            vmax=parameters['metal_frac_cbar_lim'][1],
        )
        mappable = plot_2Dhistogram('density', 'temperature', norm=norm)
        cbar_label = f"Mean (Logarithmic) metal fraction $\\log_{{10}} Z$ (min. $Z=10^{{{parameters['min_metal_frac']}}}$)"
    elif plot_name == 'density_temperature_dust_to_metal':
        norm = LogNorm(
            vmin=parameters['dust_to_metal_cbar_lim'][0], 
            vmax=parameters['dust_to_metal_cbar_lim'][1],
        )
        mappable = plot_2Dhistogram('density', 'temperature', norm=norm)
        cbar_label = f"Dust to Metal Ratio"
    elif plot_name == 'density_temperature_HI_frac':
        norm = LogNorm(
            vmin=parameters['HI_frac_cbar_lim'][0], 
            vmax=parameters['HI_frac_cbar_lim'][1],
        )
        mappable = plot_2Dhistogram('density', 'temperature', norm=norm)
        cbar_label = f"HI Mass Fraction"
    else:
        raise NotImplementedError

    # Add name of simulation to plots
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

    fig.colorbar(mappable, ax=ax, label=cbar_label)
    metadata = {f'plot_info_:{k}': str(v) for k, v in parameters.items()}
    fig.savefig(plot_name+'.png', bbox_inches='tight', metadata=metadata)

print('Done!')

