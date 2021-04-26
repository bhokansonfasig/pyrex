"""A script for combining the outputs of multiple effective volume simulations"""

import argparse
import os, os.path
import re
import numpy as np
import matplotlib.pyplot as plt

import pyrex

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',
                    help="directory containing simulation output files")
parser.add_argument('filename_pattern',
                    help=("pattern of file names to use, where 'ENERGY' will "+
                          "be replaced by the GeV energy in scientific "+
                          "notation (e.g. '1e9') and 'INDEX' will be replaced "+
                          "an incrementing index"))
args = parser.parse_args()


# Get a list of all of the files to read in
files = []
energies = []
full_pattern = re.compile(args.filename_pattern.replace("ENERGY", ".*").replace("INDEX", ".*"))
for filename in sorted(os.listdir(args.directory)):
    if full_pattern.match(filename):
        files.append(filename)
        e_start = args.filename_pattern.index("ENERGY")
        e_stop = -len(filename[e_start:].lstrip("1234567890e"))
        energy = filename[e_start:e_stop]
        if energy not in energies:
            energies.append(energy)

energies.sort(key=lambda e: float(e))

# Iterate over the files of each energy, calculating the total effective volume
veffs = []
veff_errs = []
for energy in energies:
    energy_pattern = re.compile(args.filename_pattern.replace("ENERGY", energy).replace("INDEX", ".*"))
    energy_veffs = []
    energy_veff_errs = []
    for filename in files:
        if energy_pattern.match(filename):
            with pyrex.File(os.path.join(args.directory, filename)) as file:
                energy_veffs.append(file['effective_volume'][0][0])
                energy_veff_errs.append(file['effective_volume'][0][1])
    # Note that averaging the effective volumes like this is only valid when
    # each file (in a given energy) has the same number of events thrown as
    # well as the same generation volume. If these conditions aren't met, each
    # event in the files would need to be iterated in order to calculate the
    # effective volume for the files as a whole.
    veffs.append(np.mean(energy_veffs))
    veff_errs.append(np.sqrt(np.sum(np.array(energy_veff_errs)**2))/len(energy_veff_errs))


energies = np.array([float(e) for e in energies])
veffs = np.array(veffs)/1e9
veff_errs = np.array(veff_errs)/1e9

print("Energy [GeV] \tVeff [km^3] \tError [km^3]")
for energy, veff, err in zip(energies, veffs, veff_errs):
    print(f"{energy:.0e}          \t{veff:.5e} \t{err:.5e}")

plt.loglog(energies, veffs)
plt.fill_between(energies, [v-e for v, e in zip(veffs, veff_errs)],
                 [v+e for v, e in zip(veffs, veff_errs)],
                 color='C0', alpha=0.2)
plt.xlabel("Neutrino Energy [GeV]")
plt.ylabel("Effective Volume [km^3]")
plt.tight_layout()
plt.show()
