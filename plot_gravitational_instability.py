import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import swiftsimio as sw
import unyt
plt.style.use('./mnras.mplstyle')
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

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

logT_min      = 1.
logT_max      = 8.
dlogT         = 0.01

lognH_arr = np.arange(lognH_min, lognH_max + dlognH, dlognH)
logT_arr  = np.arange(logT_min , logT_max  + dlogT, dlogT)
logT_arr_large  = np.arange(logT_min , logT_max + 4.  + dlogT, dlogT)

lognH_2Darr = np.tile(lognH_arr, (len(logT_arr), 1))
logT_2Darr  = (np.tile(logT_arr, (len(lognH_arr), 1))).T

nH_2Darr = np.power(10., lognH_2Darr)
T_2Darr = np.power(10., logT_2Darr)

def add_lambdaJs_equal_lsmooth(ax, mB, lsoft, ls = 'dashed', lc = 'r', lw = 2., lf = 1.):

    kernel_size = kernel_support_over_smoothing_length * h_smooth(mB, nH_2Darr, 0.0)

    eps = lsoft/softening_length_over_eps
    all_zones = np.zeros_like(lognH_2Darr)
    all_zones[:] = Jeans_length_W(nH_2Darr, T_2Darr, eps) / kernel_size
    CS = ax.contour(10**lognH_2Darr, 10**logT_2Darr, all_zones, levels = [(1.)], colors = lc, linewidths = lw, linestyles = ls)

    return

fig, ax = plt.subplots(1, figsize=(5, 4), constrained_layout=False)

# Low res
add_lambdaJs_equal_lsmooth(ax, 1.5e7, 0, ls=':', lc='#87CEFA')
# z = 0
# add_lambdaJs_equal_lsmooth(ax, 1.5e7, 1400, ls='-', lc='#87CEFA')
# z = 4
add_lambdaJs_equal_lsmooth(ax, 1.5e7, 3600 / 5, ls='-', lc='#87CEFA')
ax.plot(10**2, 10**4, color='#87CEFA', label='m7')

# Mid res
add_lambdaJs_equal_lsmooth(ax, 1.8e6, 0, ls=':', lc='#FF7F50')
# z = 0
# add_lambdaJs_equal_lsmooth(ax, 1.8e6, 700, ls='-', lc='#FF7F50')
# z = 4
add_lambdaJs_equal_lsmooth(ax, 1.8e6, 1800 / 5, ls='-', lc='#FF7F50')
ax.plot(10**2, 10**4, color='#FF7F50', ls='-', label='m6')

# High res
add_lambdaJs_equal_lsmooth(ax, 2.3e5, 0, ls=':', lc='#9E0000')
# z = 0
# add_lambdaJs_equal_lsmooth(ax, 2.3e5, 350, ls='-', lc='#9E0000')
# z = 4
add_lambdaJs_equal_lsmooth(ax, 2.3e5, 900 / 5, ls='-', lc='#9E0000')
ax.plot(10**2, 10**4, color='#9E0000', ls='-', label='m5')

# Line styles
ax.plot(10**2, 10**4, 'k-', label='$M_{J,soft} = <N_{ngb}> m_g$')
ax.plot(10**2, 10**4, 'k:', label='$M_J = <N_{ngb}> m_g$')

########### Adding simulation data

colibre_dir = '/cosma8/data/dp004/colibre/Runs'
# run, snap_nr = 'L025_m7/THERMAL_AGN_m7', 127
# run, snap_nr = 'L100_m6/THERMAL_AGN_m6', 127
run, snap_nr = 'L100_m6/THERMAL_AGN_m6', 56
snap = sw.load(f'{colibre_dir}/{run}/snapshots/colibre_{snap_nr:04}/colibre_{snap_nr:04}.hdf5')

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
fig.colorbar(mappable, ax=ax, label='Mass fraction')

handles, labels = ax.get_legend_handles_labels()
legend1 = ax.legend(handles[:3], labels[:3], loc='upper right')
legend2 = ax.legend(handles[3:], labels[3:], loc='upper left')
ax.add_artist(legend1)
ax.add_artist(legend2)

ax.set_xlim(density_bounds)
ax.set_ylim(temperature_bounds)
ax.loglog()
ax.set_xlabel(r"Density [$n_\mathrm{H}$ cm$^{-3}$]")
ax.set_ylabel("Temperature [K]")
outputname = run.replace('/', '_') + f'_s{snap_nr}_grav_instability.png'
plt.subplots_adjust(left=0.14, right=0.95, top=0.9, bottom=0.15)
fig.savefig(outputname, dpi = 250)
plt.close()

