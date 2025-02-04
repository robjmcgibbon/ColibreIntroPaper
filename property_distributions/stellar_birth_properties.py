"""
Plots the distribution of various stellar birth properties.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import unyt
import swiftsimio as sw
plt.style.use('../mnras.mplstyle')

# Arguments passed when running the script
parser = argparse.ArgumentParser()
# TODO
#base_dir = f'/cosma8/data/dp004/colibre/Runs'
base_dir = f'/net/hypernova/data2/COLIBRE'
parser.add_argument('--sims', nargs='+', type=str, required=True, help="Simulation names")
args = parser.parse_args()

# Define plot parameters
number_of_bins = 256

# Define bins, units, xlabel, ylabel
prop_info =  {
    'densities': (
        unyt.unyt_array(np.logspace(-2, 7, number_of_bins), units="1/cm**3"),
        lambda arr: arr.to("g/cm**3") / unyt.mh.to("g").value,
        "Stellar Birth Density $\\rho_B$ [$n_{\\rm H}$ cm$^{-3}$]",
        "$N_{\\rm bin}$ / d$\\log\\rho_B$ / $N_{\\rm total}$",
    ),
    'temperatures': (
        unyt.unyt_array(np.logspace(1, 4.5, number_of_bins), units="K"),
        lambda arr: arr.to("K").value,
        "Stellar Birth Temperature $T_B$ [K]",
        "$N_{\\rm bin}$ / d$\\log(T_B)$ / $N_{\\rm total}$"
    ),
    'velocity_dispersions': (
        unyt.unyt_array(np.logspace(0.25, 5.25, number_of_bins), units="km**2/s**2"),
        lambda arr: arr.to("km**2/s**2").value,
        "Stellar Birth Velocity Dispersion $\\sigma{}_B^2$ [km$^2$ s$^{-2}$]",
        "$N_{\\rm bin}$ / d$\\log(\\sigma{}_B^2)$ / $N_{\\rm total}$"
    ),
}


for prop, (bins, convert, xlabel, ylabel) in prop_info.items():

    log_bin_width = np.log10(bins[1].value) - np.log10(bins[0].value)
    centres = 0.5 * (bins[1:] + bins[:-1])

    fig, ax = plt.subplots(1, figsize=(5, 4), constrained_layout=False)
    plt.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.12)
    ax.loglog()

    for sim in args.sims:
        snapshot_filename = f'{base_dir}/{sim}/snapshots/colibre_0127/colibre_0127.hdf5'
        snap = sw.load(snapshot_filename)

        birth_prop = getattr(snap.stars, f'birth_{prop}')
        birth_prop = convert(birth_prop)
        n_star = birth_prop.shape[0]

        H, _ = np.histogram(birth_prop, bins=bins.value)
        y_points = H / log_bin_width / n_star

        label = sim
        ax.plot(centres, y_points, label=label)

    ax.legend(loc="upper right", markerfirst=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.savefig(f"birth_{prop}_distribution.pdf")
    plt.close()
