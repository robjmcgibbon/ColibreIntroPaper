import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
import swiftsimio as sw
import unyt
import helpers

# Matplotlib setup
TICK_LENGTH_MAJOR = 9
TICK_LENGTH_MINOR = 5
TICK_WIDTH = 1.7
PLOT_SIZE = 8
LABEL_SIZE = 30
LEGEND_SIZE = 21
LINE_WIDTH = 4
plt.style.use('./mnras.mplstyle')
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["axes.linewidth"] = 2
path_effects = [pe.Stroke(linewidth=LINE_WIDTH + 0.5, foreground="k"), pe.Normal()]

def Jeans_length_0(nH,T):
    return 5.4 * np.power(T, 1./2.) * np.power(nH, -1./2.)

def Jeans_length_W(nH,T,eps):
    H = 3. * eps
    return Jeans_length_0(nH,T) * np.power(1. + 0.27 * np.power(H / Jeans_length_0(nH,T), 2.), 0.3)

def h_smooth(mB, nH, hmin):
    h_pc = 82. * np.power(mB / 1.e6, 1./3.) * np.power(nH / 100., -1./3.)
    h_pc = np.maximum(h_pc, hmin)
    return h_pc

kernel_support_over_smoothing_length = 1.936492
softening_length_over_eps = 1.5
gamma_kernel = kernel_support_over_smoothing_length

lognH_min    = -12.
lognH_max    = 8.
dlognH       = 0.01

logT_min      = 0.8
logT_max      = 8.
dlogT         = 0.01

lognH_arr = np.arange(lognH_min, lognH_max + dlognH, dlognH)
logT_arr  = np.arange(logT_min , logT_max  + dlogT, dlogT)
logT_arr_large  = np.arange(logT_min , logT_max + 4.  + dlogT, dlogT)

lognH_2Darr = np.tile(lognH_arr, (len(logT_arr), 1))
logT_2Darr  = (np.tile(logT_arr, (len(lognH_arr), 1))).T

nH_2Darr = np.power(10., lognH_2Darr)
T_2Darr = np.power(10., logT_2Darr)

def add_lambdaJs_equal_lsmooth(ax, mB, lsoft, ls = 'dashed', lc = 'r', lw = LINE_WIDTH, lf = 1.):

    kernel_size = kernel_support_over_smoothing_length * h_smooth(mB, nH_2Darr, 0.0)

    eps = lsoft/softening_length_over_eps
    all_zones = np.zeros_like(lognH_2Darr)
    all_zones[:] = Jeans_length_W(nH_2Darr, T_2Darr, eps) / kernel_size
    CS = ax.contour(10**lognH_2Darr, 10**logT_2Darr, all_zones, levels = [(1.)], colors = lc, linewidths = lw, linestyles = ls)

    for collection in CS.collections:
        collection.set_path_effects(path_effects)
    return

# Functions above were provided by Sylvia

for run, snap_nr in [
    ('L025_m7/THERMAL_AGN_m7', 127),
    ('L025_m7/THERMAL_AGN_m7', 56),
    # ('L100_m6/THERMAL_AGN_m6', 127),
    # ('L100_m6/THERMAL_AGN_m6', 56),
    ]:
    colibre_dir = '/cosma8/data/dp004/colibre/Runs'
    snap = sw.load(f'{colibre_dir}/{run}/snapshots/colibre_{snap_nr:04}/colibre_{snap_nr:04}.hdf5')
    redshift = int(round(snap.metadata.redshift, 0))

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

    # Low res
    _, color, _ = helpers.get_sim_plot_style('L025m7/THERMAL')
    add_lambdaJs_equal_lsmooth(ax, 1.5e7, 0, ls=':', lc=color)
    lsoft = {127: 1400, 56: 3600 / 5}[snap_nr]
    add_lambdaJs_equal_lsmooth(ax, 1.5e7, lsoft, ls='-', lc=color)
    ax.plot(10**2, 10**4, color=color, label='m7', lw=LINE_WIDTH, path_effects=path_effects)

    # Mid res
    _, color, _ = helpers.get_sim_plot_style('L025m6/THERMAL')
    add_lambdaJs_equal_lsmooth(ax, 1.8e6, 0, ls=':', lc=color)
    lsoft = {127: 700, 56: 1800 / 5}[snap_nr]
    add_lambdaJs_equal_lsmooth(ax, 1.8e6, lsoft, ls='-', lc=color)
    ax.plot(10**2, 10**4, color=color, label='m6', lw=LINE_WIDTH, path_effects=path_effects)

    # High res
    _, color, _ = helpers.get_sim_plot_style('L025m5/THERMAL')
    add_lambdaJs_equal_lsmooth(ax, 2.3e5, 0, ls=':', lc=color)
    lsoft = {127: 350, 56: 900 / 5}[snap_nr]
    add_lambdaJs_equal_lsmooth(ax, 2.3e5, lsoft, ls='-', lc=color)
    ax.plot(10**2, 10**4, color=color, ls='-', label='m5', lw=LINE_WIDTH, path_effects=path_effects)

    # Line styles
    ax.plot(10**2, 10**4, 'k-', label='$M_{J,soft} = <N_{ngb}> m_g$', lw=LINE_WIDTH)
    ax.plot(10**2, 10**4, 'k:', label='$M_J = <N_{ngb}> m_g$', lw=LINE_WIDTH)

    ########### Adding simulation data

    nh = np.log10((snap.gas.densities.to_physical() / unyt.mh).to(unyt.cm ** -3).value)
    T = np.log10(snap.gas.temperatures.to_physical().to("K").value)
    total_mass = np.sum(snap.gas.masses)
    mass_fraction = (snap.gas.masses / total_mass).value

    n_bin = 256
    density_bounds = np.array([10 ** (-9.5), 1e8])            # nh/cm^3
    temperature_bounds = np.array([10 ** (1), 10 ** (12)])    # K
    x_bins = np.linspace(
        np.log10(density_bounds[0]), 
        np.log10(density_bounds[1]), 
        n_bin,
    )
    y_bins = np.linspace(
        np.log10(temperature_bounds[0]), 
        np.log10(temperature_bounds[1]), 
        n_bin,
    )
    H_norm, x_edges, y_edges = np.histogram2d(
        nh, 
        T,
        bins=[x_bins, y_bins],
        weights=mass_fraction
    )
    hist = H_norm.T

    vmin = 10**-9
    vmax = 3 * 10**-3
    norm = LogNorm(vmin=vmin, vmax=vmax)
    mappable = ax.pcolormesh(
        10**x_edges, 
        10**y_edges, 
        np.ma.array(hist, mask=(hist==0)), 
        norm=norm
    )
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(label='Mass fraction', fontsize=LABEL_SIZE)
    cbar.ax.tick_params(which="both", width=TICK_WIDTH)
    cbar.ax.tick_params(which="major", length=TICK_LENGTH_MAJOR)
    cbar.ax.tick_params(which="minor", length=TICK_LENGTH_MINOR)
    cbar.ax.tick_params(labelsize=LABEL_SIZE)

    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles[:3], labels[:3], loc='upper right', fontsize=LEGEND_SIZE)
    legend2 = ax.legend(handles[3:], labels[3:], loc='upper left', fontsize=LEGEND_SIZE)
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.set_title(f'$z={redshift}$', fontsize=LABEL_SIZE)

    ax.set_xlim(density_bounds)
    ax.set_ylim(temperature_bounds)
    ax.loglog()
    ax.set_xlabel(r"Density [$n_\mathrm{H}$ cm$^{-3}$]", fontsize=LABEL_SIZE)
    ax.set_ylabel("Temperature [K]", fontsize=LABEL_SIZE)
    outputname = run.replace('/', '_') + f'_s{snap_nr}_grav_instability.png'
    fig.savefig(outputname, dpi = 250)
    plt.close()

