##### TESTING
python cosmic_SNIa_rate.py --sim L025_m7/THERMAL_AGN_m7
python phase_diagrams.py --sim L025_m7/THERMAL_AGN_m7 --snap-nr 127 --generate-data
python phase_diagrams.py --sim L025_m7/THERMAL_AGN_m7 --snap-nr 127
python plot_pdf.py --sims L025_m7/THERMAL_AGN_m7

##### BIG BOXES
# python cosmic_SNIa_rate.py --sim L200_m7/THERMAL_AGN_m7 L200_m7/HYBRID_AGN_m7 L050_m6/THERMAL_AGN_m6 L050_m6/HYBRID_AGN_m6 L025_m5/THERMAL_AGN_m5
# python phase_diagrams.py --sim L0100N1504/Thermal --snap-nr 127 --generate-data
# python phase_diagrams.py --sim L0100N1504/Thermal --snap-nr 127
# python plot_pdf.py --sims L025_m7/THERMAL_AGN_m7 L025_m6/THERMAL_AGN_m6 L025_m5/THERMAL_AGN_m5
# python plot_gravitational_instability.py

# NOTE: You need to changes the simulations within plot_gravitational_instability.py
# NOTE: L200m6 may run out of memory if all phase plots are generated at once
