import h5py
import numpy as np
import swiftsimio as sw

############# Output list of snapshots/snipshots

run_dir = '/cosma8/data/dp004/colibre/Runs/L0012N0094/Thermal/'
with open(run_dir+'output_list.txt', 'r') as file:
    outputs = file.readlines()[1:]
with open('output_list.txt', 'w') as file:
    file.write('# Output number, Redshift, Output type\n')
    for i, line in enumerate(outputs):
        z = float(line.split(',')[0])
        t = line.split(',')[1]
        file.write(f'{i:03}, ' + f'{z:.4f}'.rjust(7) + f',{t}')

############# Output properties for each run type

corrections = {
    'SpeciesFractions': r'Species fractions array for all ions and molecules in the CHIMES network. The fraction of species i is defined in terms of its number density relative to hydrogen, i.e. $n_i$ / $n_{H_{tot}}$.',
    'VelocityDivergenceTimeDifferentials': r'Time differential (over the previous step) of the velocity divergence field around the particles. Again, provided without cosmology as this includes a Hubble flow term.',
    'VelocityDivergences': r'Local velocity divergence field around the particles. Provided without cosmology, as this includes the Hubble flow.',
    'KickedByJetFeedback': r'Flags the particles that have been directly kicked by an AGN jet feedback event at some point in the past. If greater than 0, contains the number of individual events',
    'GasVelocityDispersions': r'Velocity dispersion (3D) of the gas particles around the black holes. This is $a \sqrt{<|dx/dt|^2> - <|dx/dt|>^2}$ where x is the co-moving position of the particles relative to the black holes.',
    'MinimalSmoothingLengths': 'Minimal smoothing lengths ever reached by the particles',
}

colibre_dir = '/cosma8/data/dp004/colibre/Runs/'
snapshots = {
    'snip': f'{colibre_dir}/L100_m7/HYBRID_AGN_m7/snapshots/colibre_0126/colibre_0126.hdf5',
    'thermal': f'{colibre_dir}/L0012N0094/Thermal/snapshots/colibre_0127/colibre_0127.hdf5',
    'hybrid': f'{colibre_dir}/L100_m7/HYBRID_AGN_m7/snapshots/colibre_0127/colibre_0127.hdf5',
    'gcs': '/cosma8/data/dp004/jlvc76/COLIBRE/ScienceRuns/L0050N1504/Thermal_Ppivot1p5e4_npivot1p0/snapshots/colibre_0048/colibre_0048.hdf5',
    'gcs_snip': '/cosma8/data/dp004/jlvc76/COLIBRE/ScienceRuns/L0050N1504/Thermal_Ppivot1p5e4_npivot1p0/snapshots/colibre_0047/colibre_0047.hdf5',
}

# Requires 
#  \usepackage{longtable}
#  \usepackage{geometry}
#  \geometry{landscape}
table = r"""
\begin{longtable}{ccp{8cm}c}
\hline
\textbf{Particle type} & \textbf{Field Name} & \textbf{Description} & \textbf{Snipshots} \\
\hline
"""

for ptype, ptype_name in [
        (1, 'DM'),
        (0, 'Gas'),
        (4, 'Stars'),
        (5, 'Black holes'),
    ]:
    prop_info = {}
    # Loop through all properties in snapshot and extract description
    for snap_type, filename in snapshots.items():
        prop_info[snap_type] = {}
        names, descs = [], []
        with h5py.File(filename, 'r') as file:
            for name in file[f'PartType{ptype}'].keys():
                desc = file[f'PartType{ptype}/{name}'].attrs['Description'].decode()
                prop_info[snap_type][name] = desc

    # Add properties to table (checking if present in snipshot)
    for name, desc in prop_info['thermal'].items():
        in_snip = r"\ding{51}" if name in prop_info['snip'] else r"\ding{53}"
        desc = desc.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
        if name in corrections:
            desc = corrections[name]
        table += f"{ptype_name} & {name} & {desc} & {in_snip} \\\\\n"
    table += '\\hline \n'

    # Add hybrid properties to table
    if ptype in [0, 4, 5]:
        for name, desc in prop_info['hybrid'].items():
            if name in prop_info['thermal']:
                continue
            in_snip = r"\ding{51}" if name in prop_info['snip'] else r"\ding{53}"
            desc = desc.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
            if name in corrections:
                desc = corrections[name]
            table += f"{ptype_name} (Hybrid) & {name} & {desc} & {in_snip} \\\\\n"
        table += '\\hline \n'

    # Add globular cluster properties
    if ptype in [4]:
        for name, desc in prop_info['gcs'].items():
            if name in prop_info['thermal']:
                continue
            in_snip = r"\ding{51}" if name in prop_info['gcs_snip'] else r"\ding{53}"
            desc = desc.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
            if name in corrections:
                desc = corrections[name]
            name = name.replace('GCs_', '')
            table += f"{ptype_name} (GCs) & {name} & {desc} & {in_snip} \\\\\n"
        table += '\\hline \n'

table += r"""
\end{longtable}
"""

print(table)

