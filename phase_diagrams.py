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

import astropy
import h5py
import matplotlib.pyplot as plt
import numpy as np
import psutil
import swiftsimio as sw
import unyt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.animation import FuncAnimation

# Matplotlib setup
TICK_LENGTH_MAJOR = 9
TICK_LENGTH_MINOR = 5
TICK_WIDTH = 1.7
PLOT_SIZE = 8
LABEL_SIZE = 30
LEGEND_SIZE = 21
plt.style.use('./mnras.mplstyle')
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["axes.linewidth"] = 2

parameters = {
    # Data generation
    # These are attached as metadata to the data file
    'n_bin': 256,
    'density_bounds': np.array([10 ** (-9.5), 1e7]),            # nh/cm^3
    'temperature_bounds': np.array([10 ** (0), 10 ** (9.5)]),   # K
    'pressure_bounds': np.array([10 ** (-8.0), 10 ** 16.0]),    # K/cm^1
    # Plotting only
    'density_xlim': np.array([10 ** (-9.5), 1e7]),              # nh/cm^3
    'temperature_ylim': np.array([10 ** (0), 10 ** (9.5)]),     # K
    'pressure_ylim': np.array([10 ** (-8.0), 10 ** 12.0]),      # K/cm^1
    'metallicity_cbar_lim': np.array([-3, 0.5]),                # dimensionless (log)
    'solar_metal_frac_cbar_lim': np.array([1e-3, 2e0]),       # dimensionless
    'dust_to_metal_cbar_lim': np.array([1e-2, 7e-1]),           # dimensionless
    'small_to_large_cbar_lim': np.array([1e-2, 1e0]),           # dimensionless
    'HI_frac_cbar_lim': np.array([1e-3, 1]),                    # dimensionless
    'H2_frac_cbar_lim': np.array([1e-3, 1]),                    # dimensionless
    '100Myr_feedback_frac_cbar_lim': np.array([1e-3, 1]),       # dimensionless
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
    'density_temperature_solar_metal_frac': [
        'n_bin',
        'density_bounds',
        'temperature_bounds',
    ],
    'density_temperature_dust_to_metal': [
        'n_bin',
        'density_bounds',
        'temperature_bounds',
    ],
    'density_temperature_small_to_large': [
        'n_bin',
        'density_bounds',
        'temperature_bounds',
    ],
    'density_temperature_HI_frac': [
        'n_bin',
        'density_bounds',
        'temperature_bounds',
    ],
    'density_temperature_H2_frac': [
        'n_bin',
        'density_bounds',
        'temperature_bounds',
    ],
    'density_temperature_100Myr_feedback_frac': [
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
    elif dataset_name == 'small_dust_frac':
        dfracs = np.zeros_like(snap.gas.masses.value)
        for col in snap.gas.dust_mass_fractions.named_columns:
            if 'Small' in col:
                dfracs += getattr(snap.gas.dust_mass_fractions, col).value
        datasets[dataset_name] = dfracs
    elif dataset_name == 'large_dust_frac':
        dfracs = np.zeros_like(snap.gas.masses.value)
        for col in snap.gas.dust_mass_fractions.named_columns:
            if 'Large' in col:
                dfracs += getattr(snap.gas.dust_mass_fractions, col).value
        datasets[dataset_name] = dfracs
    elif dataset_name == 'small_dust_mass':
        datasets[dataset_name] = load_dataset('small_dust_frac') * load_dataset('mass')
    elif dataset_name == 'large_dust_mass':
        datasets[dataset_name] = load_dataset('large_dust_frac') * load_dataset('mass')
    elif dataset_name == 'dust_mass':
        datasets[dataset_name] = load_dataset('small_dust_mass') + load_dataset('large_dust_mass')
    # Metals
    elif dataset_name == 'metal_frac':
        datasets[dataset_name] = snap.gas.metal_mass_fractions.value
    elif dataset_name == 'solar_metal_frac':
        solar_metal_frac = 0.0134
        datasets[dataset_name] = load_dataset('metal_frac') / solar_metal_frac
    elif dataset_name == 'solar_metal_mass':
        datasets[dataset_name] = load_dataset('solar_metal_frac') * load_dataset('mass')
    elif dataset_name == 'metal_mass':
        metal_frac = load_dataset('metal_frac')
        mass = load_dataset('mass')
        datasets[dataset_name] = metal_frac * mass
    # HI / H2
    elif dataset_name == 'hydrogen_frac':
        datasets[dataset_name] = snap.gas.element_mass_fractions.hydrogen.value
    elif dataset_name == 'HI_mass':
        mfrac = snap.gas.species_fractions.HI.value * load_dataset('hydrogen_frac')
        datasets[dataset_name] = mfrac * load_dataset('mass')
    elif dataset_name == 'H2_mass':
        mfrac = 2 * snap.gas.species_fractions.H2.value * load_dataset('hydrogen_frac')
        datasets[dataset_name] = mfrac * load_dataset('mass')
    # Feedback times
    elif dataset_name == 'last_feedback_a':
        last_feedback_a = snap.gas.last_agnfeedback_scale_factors.value
        last_feedback_a = np.maximum(
            last_feedback_a,
            snap.gas.last_sniithermal_feedback_scale_factors.value,
        )
        datasets[dataset_name] = last_feedback_a
    elif dataset_name == 'feedback_in_last_100Myr':
        cosmo = snap.metadata.cosmology
        target_age = cosmo.age(0) - 100 * astropy.units.Myr
        target_z = astropy.cosmology.z_at_value(cosmo.age, target_age).value
        target_a = 1 / (1 + target_z)
        had_feedback = load_dataset('last_feedback_a') > target_a
        datasets[dataset_name] = had_feedback
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
                log=False,
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
                # Set bins with zeros for weight_2 that do have particles as 1e-100
                no_weight = H == 0.0
                H[no_weight] = 1e-100
                H_norm[no_weight] = 1.0
                hist = (H / H_norm).T

            if log:
                hist[hist != -100] = np.log10(hist[hist != -100])

            plot_data[plot_name] = {
                'hist': hist,
                f'{dataset_name_x}_edges': x_edges,
                f'{dataset_name_y}_edges': y_edges,
            }

        if plot_name == 'density_temperature':
            create_2Dhistogram(plot_name, 'density', 'temperature')
        elif plot_name == 'density_pressure':
            create_2Dhistogram(plot_name, 'density', 'pressure')
        elif plot_name == 'density_temperature_solar_metal_frac':
            create_2Dhistogram(
                plot_name, 
                'density', 
                'temperature', 
                dataset_name_weights_1='mass',
                dataset_name_weights_2='solar_metal_mass',
            )
        elif plot_name == 'density_temperature_dust_to_metal':
            create_2Dhistogram(
                plot_name, 
                'density', 
                'temperature', 
                dataset_name_weights_1='metal_mass',
                dataset_name_weights_2='dust_mass',
            )
        elif plot_name == 'density_temperature_small_to_large':
            create_2Dhistogram(
                plot_name, 
                'density', 
                'temperature', 
                dataset_name_weights_1='large_dust_mass',
                dataset_name_weights_2='small_dust_mass',
            )
        elif plot_name == 'density_temperature_HI_frac':
            create_2Dhistogram(
                plot_name, 
                'density', 
                'temperature', 
                dataset_name_weights_1='mass',
                dataset_name_weights_2='HI_mass',
            )
        elif plot_name == 'density_temperature_H2_frac':
            create_2Dhistogram(
                plot_name, 
                'density', 
                'temperature', 
                dataset_name_weights_1='mass',
                dataset_name_weights_2='H2_mass',
            )
        elif plot_name == 'density_temperature_100Myr_feedback_frac':
            create_2Dhistogram(
                plot_name, 
                'density', 
                'temperature', 
                dataset_name_weights_2='feedback_in_last_100Myr',
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


cbar_labels = {
    'density_temperature': "Number of particles",
    'density_pressure': "Number of particles",
    'density_temperature_solar_metal_frac': r"$\left<Z/\rm{Z_{\odot}}\right>$",
    'density_temperature_dust_to_metal': f"Dust-to-metal ratio",
    'density_temperature_small_to_large': f"Small-to-large dust grain mass ratio",
    'density_temperature_HI_frac': r"$\rm{H}\,\textsc{i}$ Mass Fraction",
    'density_temperature_H2_frac': r"$\rm{H_2}$ Mass Fraction",
    'density_temperature_100Myr_feedback_frac': r"Fraction received feedback (last 100 Myr)",
}

def plot_2Dhistogram(plot_data, plot_name, dataset_name_x, dataset_name_y):
    # Generate histogram
    data = plot_data[plot_name]
    cbar_lim_name = f'{plot_name.replace("density_temperature_", "")}_cbar_lim'
    if cbar_lim_name in parameters:
        vmin = parameters[cbar_lim_name][0]
        vmax = parameters[cbar_lim_name][1]
    else:
        vmin = 1
        vmax = np.max(data['hist'])
    norm = LogNorm(vmin=vmin, vmax=vmax)
    mappable = ax.pcolormesh(
        data[f'{dataset_name_x}_edges'], 
        data[f'{dataset_name_y}_edges'], 
        np.ma.array(data['hist'], mask=data['hist']==-100), 
        norm=norm
    )
    # Add minor ticks
    major_xticks = ax.get_xticks()
    minor_xticks = 10**np.arange(np.log10(major_xticks[0]), np.log10(major_xticks[-1]))
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_xticklabels([], minor=True)
    major_yticks = ax.get_yticks()
    minor_yticks = 10**np.arange(np.log10(major_yticks[0]), np.log10(major_yticks[-1]))
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_yticklabels([], minor=True)
    # Set axis limits
    ax.set_xlim(*parameters[f'{dataset_name_x}_xlim'])
    ax.set_ylim(*parameters[f'{dataset_name_y}_ylim'])
    return mappable

# Individual plots
for plot_name in plot_names:
    fig, ax = plt.subplots(1, figsize=(5/4 * PLOT_SIZE, PLOT_SIZE), constrained_layout=False)
    plt.subplots_adjust(left=0.14, right=0.95, top=0.9, bottom=0.15)
    ax.tick_params(which="both", width=TICK_WIDTH)
    ax.tick_params(which="major", length=TICK_LENGTH_MAJOR)
    ax.tick_params(which="minor", length=TICK_LENGTH_MINOR)
    ax.tick_params(
        axis="both",
        which="both",
        pad=8,
        left=True,
        right=True,
        top=True,
        bottom=True,
    )
    ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
    ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

    ax.loglog()
    ax.set_xlabel(r"Density [$n_\mathrm{H}$ cm$^{-3}$]", fontsize=LABEL_SIZE)
    if plot_name == 'density_pressure':
        mappable = plot_2Dhistogram(plot_data, plot_name, 'density', 'pressure')
        ax.set_ylabel("Pressure $P / \\mathrm{k_B}$ [K cm$^{-3}$]", fontsize=LABEL_SIZE)
    elif 'density_temperature' in plot_name:
        mappable = plot_2Dhistogram(plot_data, plot_name, 'density', 'temperature')
        ax.set_ylabel("Temperature [K]", fontsize=LABEL_SIZE)
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

    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(label=cbar_labels[plot_name], fontsize=LABEL_SIZE)
    cbar.ax.tick_params(which="both", width=TICK_WIDTH)
    cbar.ax.tick_params(which="major", length=TICK_LENGTH_MAJOR)
    cbar.ax.tick_params(which="minor", length=TICK_LENGTH_MINOR)
    cbar.ax.tick_params(labelsize=LABEL_SIZE)

    metadata = {f'plot_info_:{k}': str(v) for k, v in parameters.items()}
    fig.savefig(plot_name+'.png', metadata=metadata)
    plt.close()

# Combined plots
fig, axs = plt.subplots(1, 2, figsize=(10/4 * PLOT_SIZE, PLOT_SIZE), constrained_layout=False)
plt.subplots_adjust(
    left=0.07, right=0.975, wspace=0.23,
    top=0.9, bottom=0.15, hspace=0.15,
)
for i_plot, plot_name in enumerate([
        'density_temperature',
        'density_pressure',
    ]):
    ax = axs[i_plot]
    ax.tick_params(which="both", width=TICK_WIDTH)
    ax.tick_params(which="major", length=TICK_LENGTH_MAJOR)
    ax.tick_params(which="minor", length=TICK_LENGTH_MINOR)
    ax.tick_params(
        axis="both",
        which="both",
        pad=8,
        left=True,
        right=True,
        top=True,
        bottom=True,
    )
    ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
    ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)
    ax.loglog()
    ax.set_xlabel(r"Density [$n_\mathrm{H}$ cm$^{-3}$]", fontsize=LABEL_SIZE)
    if plot_name == 'density_pressure':
        mappable = plot_2Dhistogram(plot_data, plot_name, 'density', 'pressure')
        ax.set_ylabel("Pressure $P / \\mathrm{k_B}$ [K cm$^{-3}$]", fontsize=LABEL_SIZE)
    elif 'density_temperature' in plot_name:
        mappable = plot_2Dhistogram(plot_data, plot_name, 'density', 'temperature')
        ax.set_ylabel("Temperature [K]", fontsize=LABEL_SIZE)
    else:
        raise NotImplementedError

    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(label=cbar_labels[plot_name], fontsize=LABEL_SIZE)
    cbar.ax.tick_params(which="both", width=TICK_WIDTH)
    cbar.ax.tick_params(which="major", length=TICK_LENGTH_MAJOR)
    cbar.ax.tick_params(which="minor", length=TICK_LENGTH_MINOR)
    cbar.ax.tick_params(labelsize=LABEL_SIZE)

fig.savefig('density_temperature_pressure.png')
plt.close()

fig, axs = plt.subplots(3, 2, figsize=(10 / 4 * PLOT_SIZE, 11.5 / 4 * PLOT_SIZE), constrained_layout=False)
axs = axs.flatten()
plt.subplots_adjust(
    left=0.07, right=0.975, wspace=0.23,
    top=0.9667, bottom=0.05, hspace=0.25,
)
for i_plot, plot_name in enumerate([
        'density_temperature_solar_metal_frac',
        'density_temperature_100Myr_feedback_frac',
        'density_temperature_H2_frac',
        'density_temperature_HI_frac',
        'density_temperature_dust_to_metal',
        'density_temperature_small_to_large',
    ]):
    ax = axs[i_plot]
    ax.tick_params(which="both", width=TICK_WIDTH)
    ax.tick_params(which="major", length=TICK_LENGTH_MAJOR)
    ax.tick_params(which="minor", length=TICK_LENGTH_MINOR)
    ax.tick_params(
        axis="both",
        which="both",
        pad=8,
        left=True,
        right=True,
        top=True,
        bottom=True,
    )
    ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
    ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)
    ax.loglog()
    ax.set_xlabel(r"Density [$n_\mathrm{H}$ cm$^{-3}$]", fontsize=LABEL_SIZE)
    if plot_name == 'density_pressure':
        mappable = plot_2Dhistogram(plot_data, plot_name, 'density', 'pressure')
        ax.set_ylabel("Pressure $P / \\mathrm{k_B}$ [K cm$^{-3}$]", fontsize=LABEL_SIZE)
    elif 'density_temperature' in plot_name:
        mappable = plot_2Dhistogram(plot_data, plot_name, 'density', 'temperature')
        ax.set_ylabel("Temperature [K]", fontsize=LABEL_SIZE)
    else:
        raise NotImplementedError

    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(label=cbar_labels[plot_name], fontsize=LABEL_SIZE)
    cbar.ax.tick_params(which="both", width=TICK_WIDTH)
    cbar.ax.tick_params(which="major", length=TICK_LENGTH_MAJOR)
    cbar.ax.tick_params(which="minor", length=TICK_LENGTH_MINOR)
    cbar.ax.tick_params(labelsize=LABEL_SIZE)
fig.savefig('density_temperature_coloured.png')
plt.close()

print('Done!')

