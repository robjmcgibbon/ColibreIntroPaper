"""
Plots the distribution of various stellar birth properties.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import unyt
import swiftsimio as sw
plt.style.use('../mnras.mplstyle')

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
    return snap.stars.birth_velocity_dispersions.to("km**2/s**2").value

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
        load_stellar_birth_densities,
        unyt.unyt_array(np.logspace(-2, 7, number_of_bins), units="1/cm**3"),
        "Stellar birth density $\\rho_b$ [$n_{\\rm H}$ cm$^{-3}$]",
        "$n_{\\rm bin}$ / d$\\log\\rho_b$ / $n_{\\rm total}$",
    ),
    'birth_temperatures': (
        load_stellar_birth_temperatures,
        unyt.unyt_array(np.logspace(1, 4.5, number_of_bins), units="K"),
        "Stellar birth temperature $t_b$ [k]",
        "$n_{\\rm bin}$ / d$\\log(t_b)$ / $n_{\\rm total}$",
    ),
    'birth_velocity_dispersions': (
        load_stellar_birth_velocity_dispersions,
        unyt.unyt_array(np.logspace(0.25, 5.25, number_of_bins), units="km**2/s**2"),
        "Stellar birth velocity dispersion $\\sigma{}_b^2$ [km$^2$ s$^{-2}$]",
        "$n_{\\rm bin}$ / d$\\log(\\sigma{}_b^2)$ / $n_{\\rm total}$",
    ),
    'densities_at_last_supernova_event': (
        load_snii_gas_densities,
        unyt.unyt_array(np.logspace(-5, 7, number_of_bins), units="1/cm**3"),
        "Density of the gas heated by CCSN $\\rho_{\\rm CCSN}$ [$n_{\\rm H}$ cm$^{-3}$]",
        "$n_{\\rm bin}$ / d$\\log\\rho_{\\rm CCSN}$ / $n_{\\rm total}$",
    ),
    'densities_at_last_agn_event': (
        load_agn_gas_densities,
        unyt.unyt_array(np.logspace(-5, 7, number_of_bins), units="1/cm**3"),
        "Density of the gas heated by AGN $\\rho_{\\rm AGN}$ [$n_{\\rm H}$ cm$^{-3}$]",
        "$n_{\\rm bin}$ / d$\\log\\rho_{\\rm AGN}$ / $n_{\\rm total}$",
    ),
}


for name, (load_prop, bins, xlabel, ylabel) in prop_info.items():
    print(f'Loading and plotting {name}')

    log_bin_width = np.log10(bins[1].value) - np.log10(bins[0].value)
    centres = 0.5 * (bins[1:] + bins[:-1])

    fig, ax = plt.subplots(1, figsize=(5, 4), constrained_layout=False)
    plt.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.12)
    ax.loglog()

    for sim in args.sims:
        print(sim)
        snapshot_filename = f'{base_dir}/{sim}/snapshots/colibre_0127/colibre_0127.hdf5'
        if not os.path.exists(snapshot_filename):
            snapshot_filename = f'{base_dir}/{sim}/snapshots/colibre_0123/colibre_0123.hdf5'
        snap = sw.load(snapshot_filename)
        prop = load_prop(snap)
        n_star = prop.shape[0]

        H, _ = np.histogram(prop, bins=bins.value)
        y_points = H / log_bin_width / n_star

        label, color, ls = helpers.get_sim_plot_style(sim)
        ax.plot(centres, y_points, label=label, color=color, ls=ls)

    ax.legend(loc="upper right", markerfirst=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=1e-3)

    fig.savefig(f"{name}_distribution.pdf")
    plt.close()
