import h5py
import numpy as np
import swiftsimio as sw

############# Output list of snapshots/snipshots

run_dir = '/cosma8/data/dp004/colibre/Runs/L0012N0094/Thermal/'
with open(run_dir+'output_list.txt', 'r') as file:
    outputs = file.readlines()[1:]
with open('output_list.txt', 'w') as file:
    for i, line in enumerate(outputs):
        z = float(line.split(',')[0])
        t = line.split(',')[1]
        file.write(f'{i:03}, ' + f'{z:.4f}'.rjust(7) + f',{t}')

############# Output properties for each run type

colibre_dir = '/cosma8/data/dp004/colibre/Runs/'
thermal_dir = f'{colibre_dir}/L0012N0094/Thermal/'
hybrid_dir = f'{colibre_dir}/L100_m7/HYBRID_AGN_m7/'

# TODO: GCs
snapshots = {
    'snip': f'{hybrid_dir}snapshots/colibre_0126/colibre_0126.hdf5',
    'thermal': f'{thermal_dir}snapshots/colibre_0127/colibre_0127.hdf5',
    'hybrid': f'{hybrid_dir}snapshots/colibre_0127/colibre_0127.hdf5',
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
    for snap_type, filename in snapshots.items():
        prop_info[snap_type] = {}
        names, descs = [], []
        with h5py.File(filename, 'r') as file:
            for name in file[f'PartType{ptype}'].keys():
                desc = file[f'PartType{ptype}/{name}'].attrs['Description'].decode()
                prop_info[snap_type][name] = desc

    for name, desc in prop_info['thermal'].items():
        in_snip = r"\ding{51}" if name in prop_info['snip'] else r"\ding{53}"
        desc = desc.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
        table += f"{ptype_name} & {name} & {desc} & {in_snip} \\\\\n"
    table += '\\hline \n'

    if ptype in [0, 4, 5]:
        for name, desc in prop_info['hybrid'].items():
            if name in prop_info['thermal']:
                continue
            in_snip = r"\ding{51}" if name in prop_info['snip'] else r"\ding{53}"
            desc = desc.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
            table += f"{ptype_name} (Hybrid) & {name} & {desc} & {in_snip} \\\\\n"
        table += '\\hline \n'

table += r"""
\end{longtable}
"""

print(table)

