"""
Plots the distribution of various stellar birth properties.
"""

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import unyt
import swiftsimio as sw
plt.style.use('./mnras.mplstyle')

import helpers

# Arguments passed when running the script
parser = argparse.ArgumentParser()
base_dir = f'/cosma8/data/dp004/colibre/Runs'
parser.add_argument('--sims', nargs='+', type=str, required=True, help="Simulation names")
args = parser.parse_args()

# Define plot parameters
number_of_bins = 256

def load_stellar_birth_densities(snap):
    return (snap.stars.birth_densities.to("g/cm**3") / unyt.mh.to("g")).value

def load_stellar_birth_temperatures(snap):
    return snap.stars.birth_temperatures.to("K").value

def load_stellar_birth_velocity_dispersions(snap):
    return np.sqrt(snap.stars.birth_velocity_dispersions.to("km**2/s**2").value)

def load_stellar_birth_densities(snap):
    return (snap.stars.birth_densities.to("g/cm**3") / unyt.mh.to("g")).value

def load_snii_gas_densities(snap):
    gas_snii_densities = (snap.gas.densities_at_last_supernova_event.to(
        "g/cm**3"
    ) / unyt.mh.to("g")).value

    gas_snii_redshifts = (
        (1 / snap.gas.last_sniithermal_feedback_scale_factors.value) - 1
    )
    gas_snia_redshifts = (
        1 / snap.gas.last_snia_thermal_feedback_scale_factors - 1
    )

    # limit only to those gas/stellar particles that were in fact heated by snii
    mask = (gas_snii_redshifts >= 0.0) & (gas_snii_redshifts < gas_snia_redshifts)
    return gas_snii_densities[mask]

def load_agn_gas_densities(snap):
    gas_AGN_densities = (snap.gas.densities_at_last_agnevent.to(
        "g/cm**3"
    ) / unyt.mh.to("g")).value

    gas_AGN_redshifts = 1 / snap.gas.last_agnfeedback_scale_factors.value - 1

    gas_AGN_heated = gas_AGN_redshifts >= 0.0

    return gas_AGN_densities[gas_AGN_heated]


# define load_dataset, bins, xlabel, ylabelk
prop_info =  {
    'birth_densities': (
        [
            (
                load_stellar_birth_densities,
                unyt.unyt_array(np.logspace(-2, 7, number_of_bins), units="1/cm**3"),
                "-",
                "",
            ),
        ],
        "Birth density $n_{\\rm H}$ [cm$^{-3}$]",
        "$n_{\\rm bin}$ / d$\\log_{10}n_{\\rm H}$ / $n_{\\rm total}$",
    ),
    'birth_temperatures': (
        [
            (
                load_stellar_birth_temperatures,
                unyt.unyt_array(np.logspace(1, 4.5, number_of_bins), units="K"),
                "-",
                "",
            ),
        ],
        "Birth temperature $T_b$ [k]",
        "$n_{\\rm bin}$ / d$\\log_{10}T_b$ / $n_{\\rm total}$",
    ),
    'birth_velocity_dispersions': (
        [
            (
                load_stellar_birth_velocity_dispersions,
                unyt.unyt_array(np.logspace(0, 2.5, number_of_bins), units="km/s"),
                "-",
                "",
            ),
        ],
        "Birth velocity dispersion $\\sigma{}_b$ [km s$^{-1}$]",
        "$n_{\\rm bin}$ / d$\\log_{10}\\sigma{}_b$ / $n_{\\rm total}$",
    ),
    'densities_at_last_supernova_event': (
        [
            (
                load_snii_gas_densities,
                unyt.unyt_array(np.logspace(-5, 7, number_of_bins), units="1/cm**3"),
                "-",
                "",
            ),
        ],
        "Density of the gas heated by CCSN $\\rho_{\\rm CCSN}$ [$n_{\\rm H}$ cm$^{-3}$]",
        "$n_{\\rm bin}$ / d$\\log_{10}\\rho_{\\rm CCSN}$ / $n_{\\rm total}$",
    ),
    'densities_at_last_agn_event': (
        [
            (
                load_agn_gas_densities,
                unyt.unyt_array(np.logspace(-5, 7, number_of_bins), units="1/cm**3"),
                "-",
                "",
            ),
        ],
        "Density of the gas heated by AGN $\\rho_{\\rm AGN}$ [$n_{\\rm H}$ cm$^{-3}$]",
        "$n_{\\rm bin}$ / d$\\log_{10}\\rho_{\\rm AGN}$ / $n_{\\rm total}$",
    ),
    'birth_ccsn_densities': (
        [
            (
                load_stellar_birth_densities,
                unyt.unyt_array(np.logspace(-2, 7, number_of_bins), units="1/cm**3"),
                "-",
                "Stellar birth",
            ),
            (
                load_snii_gas_densities,
                unyt.unyt_array(np.logspace(-5, 7, number_of_bins), units="1/cm**3"),
                "--",
                "CCSN feedback",
            ),
        ],
        "Density $n_{\\rm H}$ [cm$^{-3}$]",
        "$n_{\\rm bin}$ / d$\\log_{10}n_{\\rm H}$ / $n_{\\rm total}$",
    ),
    'birth_ccsn_agn_densities': (
        [
            (
                load_stellar_birth_densities,
                unyt.unyt_array(np.logspace(-2, 7, number_of_bins), units="1/cm**3"),
                "-",
                "Stellar birth",
            ),
            (
                load_snii_gas_densities,
                unyt.unyt_array(np.logspace(-5, 7, number_of_bins), units="1/cm**3"),
                "--",
                "CCSN feedback",
            ),
            (
                load_agn_gas_densities,
                unyt.unyt_array(np.logspace(-5, 7, number_of_bins), units="1/cm**3"),
                ":",
                "AGN feedback",
            ),
        ],
        "Density $n_{\\rm H}$ [cm$^{-3}$]",
        "$n_{\\rm bin}$ / d$\\log_{10}n_{\\rm H}$ / $n_{\\rm total}$",
    ),

}

snap_data = {}
for name, (to_plot, xlabel, ylabel) in prop_info.items():
    print(f'Loading and plotting {name}')

    fig, ax = plt.subplots(1, figsize=(5, 4), constrained_layout=False)
    plt.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.12)
    ax.loglog()
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))

    for i_prop, (load_prop, bins, ls, ls_label) in enumerate(to_plot):
        log_bin_width = np.log10(bins[1].value) - np.log10(bins[0].value)
        centres = 0.5 * (bins[1:] + bins[:-1])

        for sim in args.sims:
            print(sim)
            if sim not in snap_data:
                # Load z=0 data
                snapshot_filename = f'{base_dir}/{sim}/snapshots/colibre_0127/colibre_0127.hdf5'
                if not os.path.exists(snapshot_filename):
                    snapshot_filename = f'{base_dir}/{sim}/snapshots/colibre_0123/colibre_0123.hdf5'
                snap_data[sim] = sw.load(snapshot_filename)
            snap = snap_data[sim]

            prop = load_prop(snap)
            n_part = prop.shape[0]

            H, _ = np.histogram(prop, bins=bins.value)
            y_points = H / log_bin_width / n_part

            label, color, _ = helpers.get_sim_plot_style(sim)
            if i_prop == 0:
                ax.plot(centres, y_points, label=label, color=color, ls=ls)
            else:
                ax.plot(centres, y_points, color=color, ls=ls)

        if ls_label != "":
            ax.plot(centres[0], y_points[0], color='k', ls=ls, label=ls_label)

    ax.legend(loc="upper right", markerfirst=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=1e-3, top=1e1)

    fig.savefig(f"{name}_distribution.pdf")
    plt.close()
