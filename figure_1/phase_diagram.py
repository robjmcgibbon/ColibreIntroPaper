"""
Makes a rho-T plot. Uses the swiftsimio library.
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import swiftsimio as sw
import unyt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.animation import FuncAnimation
plt.style.use('../mnras.mplstyle')

parameters = {
    # Data generation
    # These are attached as metadata to the data file
    'density_bounds': np.array([10 ** (-9.5), 1e7]),           # in nh/cm^3
    'temperature_bounds': np.array([10 ** (0), 10 ** (9.5)]),  # in K
    'pressure_bounds': np.array([10 ** (-8.0), 10 ** 8.0]),    # in K/cm^1
    'internal_energy_bounds': np.array([10 ** (-4), 10 ** 8]), # in (km / s)^2
    'dustfrac_bounds': np.array([-5, -1]),                     # dimensionless
    'n_bin': 256,
    # Plotting only
    'density_xlim': np.array([10 ** (-9.5), 1e7]),             # in nh/cm^3
    'temperature_ylim': np.array([10 ** (0), 10 ** (9.5)]),    # in K
    'pressure_ylim': np.array([10 ** (-8.0), 10 ** 8.0]),      # in K/cm^1
    'internal_energy_ylim': np.array([10 ** (-4), 10 ** 8]),   # in (km / s)^2
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
    'density_pressure': [
        'density_bounds', 
        'pressure_bounds',
        'n_bin',
    ],
    'density_internal_energy': [
        'density_bounds', 
        'internal_energy_bounds',
        'n_bin',
    ],
    'density_temperature_dustfrac': [
        'density_bounds', 
        'temperature_bounds',
        'n_bin',
        'dustfrac_bounds',
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
        datasets[dataset_name] = (snap.gas.pressures.to_physical() / unyt.kb).to(unyt.K * unyt.cm ** -3).value
    elif dataset_name == 'internal_energy':
        datasets[dataset_name] = (snap.gas.internal_energies.to_physical()).to(unyt.km ** 2 / unyt.s ** 2).value
    elif dataset_name == 'dustfrac':
        min_dfracs = parameters['dustfrac_bounds'][0]
        dfracs = np.zeros_like(snap.gas.masses.value)
        for d in dir(snap.gas.dust_mass_fractions):
            if hasattr(getattr(snap.gas.dust_mass_fractions, d), "units"):
                dfracs += getattr(snap.gas.dust_mass_fractions, d)
        dfracs[dfracs < 10.0 ** min_dfracs] = 10.0 ** min_dfracs
        datasets[dataset_name] = np.log10(dfracs.value)
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

        def load_2Dhistogram(
                plot_name,
                dataset_name_x,
                dataset_name_y, 
                dataset_name_weights=None
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
            H_norm, x_edges, y_edges = np.histogram2d(
                x, 
                y,
                bins=[x_bins, y_bins],
            )
            if dataset_name_weights is None:
                hist = H_norm.T
            else:
                weights = load_dataset(dataset_name_weights)
                H, _, _ = np.histogram2d(
                    x, 
                    y,
                    bins=[x_bins, y_bins], 
                    weights=weights,
                )
                # Set bins with no values to -100 to avoid division by 0
                mask = H_norm == 0.0
                H[mask] = -100
                H_norm[mask] = 1.0
                hist = (H / H_norm).T

            plot_data[plot_name] = {
                'hist': hist,
                f'{dataset_name_x}_edges': x_edges,
                f'{dataset_name_y}_edges': y_edges,
            }

        if plot_name == 'density_temperature':
            load_2Dhistogram(plot_name, 'density', 'temperature')
        elif plot_name == 'density_pressure':
            load_2Dhistogram(plot_name, 'density', 'pressure')
        elif plot_name == 'density_internal_energy':
            load_2Dhistogram(plot_name, 'density', 'internal_energy')
        elif plot_name == 'density_temperature_dustfrac':
            load_2Dhistogram(
                plot_name, 
                'density', 
                'temperature', 
                dataset_name_weights='dustfrac'
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

    if plot_name == 'density_temperature':
        mappable = plot_2Dhistogram('density', 'temperature')
        ax.set_xlabel("Density [$n_H$ cm$^{-3}$]")
        ax.set_ylabel("Temperature [K]")
        cbar_label = "Number of particles"
    elif plot_name == 'density_pressure':
        mappable = plot_2Dhistogram('density', 'pressure')
        ax.set_xlabel("Density [$n_H$ cm$^{-3}$]")
        ax.set_ylabel("Pressure $P / k_B$ [K cm$^{-3}$]")
        cbar_label = "Number of particles"
    elif plot_name == 'density_internal_energy':
        mappable = plot_2Dhistogram('density', 'internal_energy')
        ax.set_xlabel("Density [$n_H$ cm$^{-3}$]")
        ax.set_ylabel("Internal Energy [km$^2$ / s$^2$]")
        cbar_label = "Number of particles"
    elif plot_name == 'density_temperature_dustfrac':
        norm = Normalize(
            vmin=parameters['dustfrac_bounds'][0], 
            vmax=parameters['dustfrac_bounds'][1],
        )
        mappable = plot_2Dhistogram('density', 'temperature', norm=norm)
        ax.set_xlabel("Density [$n_H$ cm$^{-3}$]")
        ax.set_ylabel("Temperature [K]")
        cbar_label = "Mean (Logarithmic) Dust Mass Fraction"
    else:
        raise NotImplementedError

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
    fig.savefig(plot_name+'.png')

print('Done!')

