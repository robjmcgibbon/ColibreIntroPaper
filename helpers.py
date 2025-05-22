def get_sim_plot_style(sim):
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
        raise NotImplementedError('Sim must contain m5/m6/m7 (e.g. L50_m5/THERMAL')

    if 'THERMAL' in sim:
        ls = '-'
    elif 'HYBRID' in sim:
        ls = '--'
        label += 'h'
    else:
        raise NotImplementedError('Sim must contain THERMAL/HYBRID')

    return label, color, ls


