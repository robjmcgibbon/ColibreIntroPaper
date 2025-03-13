def get_sim_plot_style(sim):
    # Extract box size
    label = sim[:5]

    if 'm5' in sim:
        color = '#cd5c5c'
        label += 'm5'
    elif 'm6' in sim:
        color = '#ffa500'
        label += 'm6'
    elif 'm7' in sim:
        color = '#00bfff'
        label += 'm7'
    else:
        raise NotImplementedError('Sim must contain m5/m6/m7 (e.g. L50_m5/THERMAL')

    if 'THERMAL' in sim:
        ls = '-'
        label += ' THERMAL'
    elif 'HYBRID' in sim:
        ls = '--'
        label += ' HYBRID'
    else:
        raise NotImplementedError('Sim must contain THERMAL/HYBRID')

    return label, color, ls


