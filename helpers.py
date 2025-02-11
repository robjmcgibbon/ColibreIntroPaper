def get_sim_plot_style(sim):
    label = sim
    if 'THERMAL' in sim:
        color = 'darkorange'
    elif 'HYBRID' in sim:
        color = 'deepskyblue'
    else:
        raise NotImplementedError('Sim must contain THERMAL/HYBRID')
    if 'm5' in sim:
        ls = ':'
    elif 'm6' in sim:
        ls = '--'
    elif 'm7' in sim:
        ls = '-'
    else:
        raise NotImplementedError('Sim must contain m5/m6/m7')
    return label, color, ls


