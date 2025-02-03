"""
Plots the cosmic SNIa rate history.

This script is not split into data generation
and plotting since it's so quick to run
"""

import argparse
import glob

import astropy.cosmology
import h5py
import matplotlib.pyplot as plt
import numpy as np
import swiftsimio as sw
import unyt
from velociraptor.observations import load_observations
plt.style.use('../mnras.mplstyle')

# Arguments passed when running the script
parser = argparse.ArgumentParser()
# TODO
#base_dir = f'/cosma8/data/dp004/colibre/Runs'
base_dir = f'/net/hypernova/data2/COLIBRE'
parser.add_argument('--sim', type=str, required=True, help="Simulation name")
args = parser.parse_args()

snapshot_filename = f'{base_dir}/{args.sim}/snapshots/colibre_0127/colibre_0127.hdf5'
SNIa_filename = f'{base_dir}/{args.sim}/SNIa.txt'

# Plot setup
fig, ax = plt.subplots(1, figsize=(5, 4), constrained_layout=False)
ax.semilogx()
# TODO:
log_multiplicative_factor = 4
multiplicative_factor = 10 ** log_multiplicative_factor
SNIa_rate_output_units = 1.0 / (unyt.yr * unyt.Mpc ** 3)

# Load data from SNIa file
data = np.loadtxt(
    SNIa_filename,
    usecols=(4, 6, 11),
    dtype=[("a", np.float32), ("z", np.float32), ("SNIa rate", np.float32)],
)

# Load units and cosmology from snapshot
snapshot = sw.load(snapshot_filename)
cosmology = snapshot.metadata.cosmology
units = snapshot.units
SNIa_rate_units = 1.0 / (units.time * units.length ** 3)

# Plot simulation data, use high z-order so it's on top of observations
scale_factor = data["a"]
SNIa_rate = (data["SNIa rate"] * SNIa_rate_units).to(SNIa_rate_output_units)
ax.plot(scale_factor, SNIa_rate.value * multiplicative_factor, zorder=10000)[0]

# Plot observational data
path_to_obs_data = f"../velociraptor-comparison-data"
observational_data = load_observations(
    sorted(glob.glob(f"{path_to_obs_data}/data/CosmicSNIaRate/*.hdf5"))
)
observation_lines = []
observation_labels = []
for obs_data in observational_data:
    observation_lines.append(
        ax.errorbar(
            obs_data.x.value,
            obs_data.y.value * multiplicative_factor,
            yerr=obs_data.y_scatter.value * multiplicative_factor,
            label=obs_data.citation,
            color='grey',
            linestyle="none",
            marker="o",
            elinewidth=0.5,
            markeredgecolor="none",
            markersize=2,
            zorder=-10,
        )
    )
    observation_labels.append(f"{obs_data.citation}")

redshift_ticks = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0])
redshift_labels = [
    "$0$",
    "$0.2$",
    "$0.5$",
    "$1$",
    "$2$",
    "$3$",
    "$5$",
    "$10$",
    "$20$",
    "$50$",
    "$100$",
]
a_ticks = 1.0 / (redshift_ticks + 1.0)

ax.set_xticks(a_ticks)
ax.set_xticklabels(redshift_labels)

# Adding legend with observational data citations
#observation_legend = ax.legend(
#    observation_lines, observation_labels, markerfirst=True, loc="center right"
#)
#ax.add_artist(observation_legend)

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

# Create second X-axis (to plot cosmic time alongside redshift)
ax2 = ax.twiny()
ax2.set_xscale("log")

# Cosmic-time ticks (in Gyr) along the second X-axis
t_ticks = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, cosmology.age(1.0e-5).value])

# To place the new ticks onto the X-axis we need to know the corresponding scale factors
a_ticks_2axis = [
    1.0 / (1.0 + astropy.cosmology.z_at_value(cosmology.age, t_tick * astropy.units.Gyr)) for t_tick in t_ticks
]

# Attach the ticks to the second X-axis
ax2.set_xticks(a_ticks_2axis)

# Format the ticks' labels
ax2.set_xticklabels(["$%2.1f$" % t_tick for t_tick in t_ticks])

# Final adjustments
ax.tick_params(axis="x", which="minor", bottom=False)
ax2.tick_params(axis="x", which="minor", top=False)

plt.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)

ax.set_ylim(0.0, 1.6)
ax.set_xlim(1.02, 0.07)
ax2.set_xlim(1.02, 0.07)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(
    f"SNIa rate [$10^{{-{log_multiplicative_factor}}}$ yr$^{{-1}}$ cMpc$^{{-3}}$]"
)
ax2.set_xlabel("Cosmic time [Gyr]")

fig.savefig(f"SNIa_rate_history.pdf")
plt.close()
