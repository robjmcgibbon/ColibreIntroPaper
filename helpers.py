def get_sim_plot_style(sim):

    special = {
        'L0025N0188/Thermal_rerun': ('L025m7', '#D12424', '-'),
        'L0025N0376/Thermal': ('L025m6', '#FF9F6E', '-'),
        'L0025N0752/Thermal': ('L025m5', '#C4E8FF', '-'),
    }
    if sim in special:
        return special[sim]

    # Extract box size
    assert sim[5] != '/', 'Run directory must be in LXXX_mY format'
    label = sim[:4]

    if 'm5' in sim:
        color = '#C4E8FF'
        label += 'm5'
    elif 'm6' in sim:
        color = '#FF9F6E'
        label += 'm6'
    elif 'm7' in sim:
        color = '#D12424'
        label += 'm7'
    else:
        raise NotImplementedError('Sim must contain m5/m6/m7 (e.g. L050_m5/THERMAL')

    if 'THERMAL' in sim:
        ls = '-'
    elif 'HYBRID' in sim:
        ls = '--'
        label += 'h'
    else:
        raise NotImplementedError('Sim must contain THERMAL/HYBRID')

    return label, color, ls


