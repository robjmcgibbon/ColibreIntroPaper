"""
Plots the distribution of various stellar birth properties.
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import LogLocator
import numpy as np
import unyt
import swiftsimio as sw
plt.style.use('./mnras.mplstyle')
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

import helpers

# Arguments passed when running the script
parser = argparse.ArgumentParser()
base_dir = f'/cosma8/data/dp004/colibre/Runs'
parser.add_argument('--sims', nargs='+', type=str, required=True, help="Simulation names")
args = parser.parse_args()

def load_stellar_birth_densities(snap):
    return (snap.stars.birth_densities.to("g/cm**3") / unyt.mh.to("g")).value

def load_stellar_birth_temperatures(snap):
    return snap.stars.birth_temperatures.to("K").value

def load_stellar_birth_densities(snap):
    return (snap.stars.birth_densities.to("g/cm**3") / unyt.mh.to("g")).value

def load_stellar_birth_velocity_dispersions(snap):
    return np.sqrt(snap.stars.birth_velocity_dispersions.to("km**2/s**2").value)

def load_stellar_thermal_velocity_dispersions(snap):
    return (13.8 * np.sqrt(snap.stars.birth_temperatures / (10**4 * unyt.K))).value

def load_stellar_velocity_dispersion_ratio(snap):
    return load_stellar_birth_velocity_dispersions(snap) / load_stellar_thermal_velocity_dispersions(snap)

def load_stellar_initial_masses(snap):
    return snap.stars.initial_masses.to('Msun').value

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

def mask_galaxy_mass(low_mass, high_mass):
    def load_masked_galaxy_masses(snap, soap):
        # Create mapping from group_nr_bound to galaxy stellar mass
        halo_catalogue_idx = soap.input_halos.halo_catalogue_index.value
        stellar_mass = soap.bound_subhalo.stellar_mass.to('Msun').value
        stellar_mass_from_halo_catalogue_idx = np.zeros(
            np.max(halo_catalogue_idx) + 1, dtype=int
        )
        stellar_mass_from_halo_catalogue_idx[halo_catalogue_idx] = stellar_mass

        # Use mapping to get host galaxy mass for each particle
        particle_group_nr_bound = snap.stars.group_nr_bound.value
        host_galaxy_stellar_mass = np.zeros(particle_group_nr_bound.shape[0])
        is_bound = particle_group_nr_bound != -1
        host_galaxy_stellar_mass[is_bound] = stellar_mass_from_halo_catalogue_idx[
            particle_group_nr_bound[is_bound]
        ]

        # Create mask based on host_galaxy mass
        mask = host_galaxy_stellar_mass >= low_mass
        mask &= host_galaxy_stellar_mass < high_mass
        return mask
    return load_masked_galaxy_masses

prop_info =  {
    # Name of plot
    'birth_densities': (
        # Properties to plot
        [
            (
                # Load function for property
                load_stellar_birth_densities,
                # Bins when creating histogram
                unyt.unyt_array(np.logspace(-2, 7, 64), units="1/cm**3"),
                # Linestyle for this property
                "-",
                # Name of property (if plotting multiple properties)
                None,
            ),
        ],
        # Masks to apply (or pass None for no masks)
        [
            (
                # Load function for mask, None to load all particles
                None,
                # Linestyle for this mask (overwrites property linestyle)
                None,
                # Name of mask
                None,
            ),
        ],
        # Create a cumulative plot?
        False,
        # x_axis_label
        "Birth density $n_{\\rm H}$ [cm$^{-3}$]",
        # y_axis_label
        "$n_{\\rm bin}$ / d$\\log_{10}n_{\\rm H}$ / $n_{\\rm total}$",
        # (x_axis_limits, x_axis_scale
        ([1e-4, 1e7], "log"),
        # (y_axis_limits, x_axis_scale
        ([1e-3, 1e1], "log"),
        # Whether to print median values and add median line to plot
        False,
    ),
    'birth_densities_mass_split': (
        [
            (
                load_stellar_birth_densities,
                unyt.unyt_array(np.logspace(-2, 7, 64), units="1/cm**3"),
                "-",
                None,
            ),
        ],
        [
            (
                mask_galaxy_mass(10**7, 10**8),
                '-',
                r'$10^{7} < M_* \rm{/} \rm{M_\odot} < 10^{8}$',
            ),
            (
                mask_galaxy_mass(10**8.75, 10**9.75),
                '--',
                r'$10^{8.75} < M_*$/$\rm{M_\odot} < 10^{9.75}$',
            ),
            (
                mask_galaxy_mass(10**10.5, 10**11.5),
                ':',
                r'$10^{10.5} < M_*$/$\rm{M_\odot} < 10^{11.5}$',
            ),
        ],
        False,
        "Birth density $n_{\\rm H}$ [cm$^{-3}$]",
        "$n_{\\rm bin}$ / d$\\log_{10}n_{\\rm H}$ / $n_{\\rm total}$",
        ([1e-4, 1e7], "log"),
        ([1e-3, 1e1], "log"),
        False,
    ),
    'birth_temperatures': (
        [
            (
                load_stellar_birth_temperatures,
                unyt.unyt_array(np.logspace(1, 4.5, 128), units="K"),
                "-",
                None,
            ),
        ],
        None,
        False,
        "Birth temperature $T$ [K]",
        "$n_{\\rm bin}$ / d$\\log_{10}T$ / $n_{\\rm total}$",
        (None, "log"),
        ([1e-3, 1e1], "log"),
        False,
    ),
    'birth_temperatures_mass_split': (
        [
            (
                load_stellar_birth_temperatures,
                unyt.unyt_array(np.logspace(1, 4.5, 128), units="K"),
                "-",
                None,
            ),
        ],
        [
            (
                mask_galaxy_mass(10**7, 10**8),
                '-',
                r'$10^{7} < M_* \rm{/} \rm{M_\odot} < 10^{8}$',
            ),
            (
                mask_galaxy_mass(10**8.75, 10**9.75),
                '--',
                r'$10^{8.75} < M_*$/$\rm{M_\odot} < 10^{9.75}$',
            ),
            (
                mask_galaxy_mass(10**10.5, 10**11.5),
                ':',
                r'$10^{10.5} < M_*$/$\rm{M_\odot} < 10^{11.5}$',
            ),
        ],
        False,
        "Birth temperature $T$ [K]",
        "$n_{\\rm bin}$ / d$\\log_{10}T$ / $n_{\\rm total}$",
        (None, "log"),
        ([1e-3, 5e1], "log"),
        False,
    ),
    'ratio_birth_velocity_dispersions': (
        [
            (
                load_stellar_velocity_dispersion_ratio,
                unyt.unyt_array(np.logspace(-2, 3, 64), units="km/s"),
                "-",
                None,
            ),
        ],
        None,
        False,
        r"Velocity dispersion ratio $r = \sigma_{\rm turb}$ / $\sigma_{\rm th}$",
        "$n_{\\rm bin}$ / d$\\log_{10}r$ / $n_{\\rm total}$",
        ([1e-1, 1e3], "log"),
        ([1e-3, 1e1], "log"),
        False,
    ),
    'ratio_birth_velocity_dispersions_mass_split': (
        [
            (
                load_stellar_velocity_dispersion_ratio,
                unyt.unyt_array(np.logspace(-2, 3, 64), units="km/s"),
                "-",
                None,
            ),
        ],
        [
            (
                mask_galaxy_mass(10**7, 10**8),
                '-',
                r'$10^{7} < M_*$/$\rm{M_\odot} < 10^{8}$',
            ),
            (
                mask_galaxy_mass(10**8.75, 10**9.75),
                '--',
                r'$10^{8.75} < M_*$/$\rm{M_\odot} < 10^{9.75}$',
            ),
            (
                mask_galaxy_mass(10**10.5, 10**11.5),
                ':',
                r'$10^{10.5} < M_*$/$\rm{M_\odot} < 10^{11.5}$',
            ),
        ],
        False,
        r"Velocity dispersion ratio $r = \sigma_{\rm turb}$ / $\sigma_{\rm th}$",
        "$n_{\\rm bin}$ / d$\\log_{10}r$ / $n_{\\rm total}$",
        ([1e-1, 1e3], "log"),
        ([1e-3, 1e1], "log"),
        False,
    ),
    'ccsn_agn_densities': (
        [
            (
                load_snii_gas_densities,
                unyt.unyt_array(np.logspace(-5, 7, 64), units="1/cm**3"),
                "-",
                "CCSN feedback",
            ),
            (
                load_agn_gas_densities,
                unyt.unyt_array(np.logspace(-5, 7, 64), units="1/cm**3"),
                "--",
                "AGN feedback",
            ),
        ],
        None,
        False,
        "Density $n_{\\rm H}$ [cm$^{-3}$]",
        "$n_{\\rm bin}$ / d$\\log_{10}n_{\\rm H}$ / $n_{\\rm total}$",
        ([1e-4, 1e7], "log"),
        ([1e-3, 1e1], "log"),
        False,
    ),
}

snap_data = {}
soap_data = {}
for name, (to_plot, masks, cumulative, xlabel, ylabel, xaxis, yaxis, plot_median) in prop_info.items():
    print(f'Loading and plotting {name}')

    fig, ax = plt.subplots(1, figsize=(5, 4), constrained_layout=False)
    plt.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.12)
    path_effects = [pe.Stroke(linewidth=1.5, foreground="k"), pe.Normal()]
    lw = 1

    # Store objects we want to appear in the legends
    sim_legend_lines = []
    mask_legend_lines = []
    prop_legend_lines = []

    for i_prop, (load_prop, bins, prop_ls, prop_label) in enumerate(to_plot):

        log_bin_width = np.log10(bins[1].value) - np.log10(bins[0].value)
        centres = 0.5 * (bins[1:] + bins[:-1])

        if masks is None:
            masks = [(None, None, None)]

        for i_sim, sim in enumerate(args.sims):
            if sim not in snap_data:
                # Load z=0 data
                snapshot_filename = f'{base_dir}/{sim}/SOAP/colibre_with_SOAP_membership_0127.hdf5'
                if not os.path.exists(snapshot_filename):
                    snapshot_filename = f'{base_dir}/{sim}/SOAP/colibre_with_SOAP_membership_0123.hdf5'
                snap_data[sim] = sw.load(snapshot_filename)
            snap = snap_data[sim]

            if masks[0] is not None:
                if sim not in soap_data:
                    # Load z=0 data
                    soap_filename = f'{base_dir}/{sim}/SOAP/halo_properties_0127.hdf5'
                    if not os.path.exists(soap_filename):
                        soap_filename = f'{base_dir}/{sim}/SOAP/halo_properties_0123.hdf5'
                    snapshot_filename = f'{base_dir}/{sim}/SOAP/colibre_with_SOAP_membership_0123.hdf5'
                    soap_data[sim] = sw.load(soap_filename)
                soap = soap_data[sim]

            plot_prop_label = True
            for i_mask, (load_mask, mask_ls, mask_label) in enumerate(masks):
                prop = load_prop(snap)
                n_part = prop.shape[0]

                # Mask values if required
                if load_mask is not None:
                    mask = load_mask(snap, soap)
                    prop = prop[mask]

                # Create histogram
                H, _ = np.histogram(prop, bins=bins.value)
                if cumulative:
                    y_points = np.cumsum(H) / n_part
                else:
                    y_points = H / log_bin_width / n_part

                # Add label to indicate sim name
                label, color, _ = helpers.get_sim_plot_style(sim)
                if (i_prop == 0) & (i_mask == 0):
                    line, = ax.plot(np.mean(centres), np.mean(y_points), label=label, color=color, ls=prop_ls, path_effects=path_effects, lw=lw)
                    sim_legend_lines.append(line)

                if load_mask is not None:
                    ax.plot(centres, y_points, color=color, ls=mask_ls, path_effects=path_effects, lw=lw)
                    if (i_prop == 0) & (i_sim == 0):
                        line, = ax.plot(centres[0], y_points[0], color='k', ls=mask_ls, label=mask_label, path_effects=path_effects, lw=lw)
                        mask_legend_lines.append(line)
                    plot_prop_label = False
                else:
                    ax.plot(centres, y_points, color=color, ls=prop_ls, path_effects=path_effects, lw=lw)

                if plot_median:
                    ax.axvline(np.median(prop), color=color, ls=prop_ls, path_effects=path_effects, lw=lw)
                    print(f'{sim} {prop_label} median: {np.median(prop):.3g}')

        if plot_prop_label and (prop_label is not None):
            line, = ax.plot(centres[0], y_points[0], color='k', ls=prop_ls, label=prop_label, path_effects=path_effects, lw=lw)
            prop_legend_lines.append(line)

    legend = ax.legend(handles=sim_legend_lines, loc="upper right", markerfirst=False)
    ax.add_artist(legend)
    if len(mask_legend_lines + prop_legend_lines) != 0:
        legend = ax.legend(handles=mask_legend_lines+prop_legend_lines, loc="upper left", markerfirst=False)
        ax.add_artist(legend)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xaxis[0] is not None:
        ax.set_xlim(left=xaxis[0][0], right=xaxis[0][1])
    if yaxis[0] is not None:
        ax.set_ylim(bottom=yaxis[0][0], top=yaxis[0][1])
    if xaxis[1] == "log":
        ax.set_xscale('log')
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    if yaxis[1] == "log":
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))

    fig.savefig(f"{name}_distribution.pdf")
    plt.close()
