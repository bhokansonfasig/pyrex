"""A script for simulating effective volumes of ARA stations"""

import argparse
import datetime
import logging
import numpy as np

import pyrex
import pyrex.custom.ara as ara

logging.basicConfig(level=logging.WARNING)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('energy', type=float,
                    help="neutrino energy (in GeV)")
parser.add_argument('detector', type=int,
                    help="detector selection from list in script")
parser.add_argument('-n', '--number', type=int, default=10,
                    help="number of events to generate (default 10)")
parser.add_argument('--shadow', action='store_true',
                    help=("if present, Earth shadowing will remove events "+
                          "rather than weighting their survival probability"))
parser.add_argument('--secondaries', action='store_true',
                    help=("if present, secondary interactions will be "+
                          "considered"))
parser.add_argument('--noiseless', action='store_true',
                    help="if present, noise not added to signal")
parser.add_argument('--montecarlo', action='store_true',
                    help=("if present, suppress noise triggers based on "+
                          "MC truth"))
parser.add_argument('-o', '--output', required=True,
                    help="output file name")
args = parser.parse_args()


# Prepare the output file. In this case, we'll record the waveforms on the
# antennas for any triggering events. This will create very large files,
# so unless recording the waveforms is crucial, it would be better to change
# write_waveforms to False.
outfile = pyrex.File(args.output, 'w',
                     write_particles=True,
                     write_waveforms=True,
                     write_triggers=True,
                     write_antenna_triggers=True,
                     write_rays=True,
                     write_noise=False,
                     require_trigger=['waveforms'])
outfile.open()

# Add an additional metadata location in the output file for recording the
# parameters used by this run of the simulation.
outfile.create_analysis_metadataset("sim_parameters")

# Dictionary for metadata from the script for the output file. This will be
# written to the "sim_parameters" dataset once it is filled.
script_metadata = {}


# Set up detector and trigger, based on the script's argument values
def pol_trigger(det, ant_type, require_mc_truth=False):
    if require_mc_truth:
        return sum(1 for ant in det
                   if isinstance(ant, ant_type)
                   and ant.is_hit_mc_truth)
    else:
        return sum(1 for ant in det
                   if isinstance(ant, ant_type)
                   and ant.is_hit)

if args.detector==2:
    detector = ara.ARA02()
    detector.build_antennas(power_threshold=6.46, noisy=not(args.noiseless))
    is_triggered = {
        "global": lambda detector: detector.triggered(
            polarized_antenna_requirement=3,
            require_mc_truth=args.montecarlo,
        ),
        "vpol": lambda detector: pol_trigger(
            det=detector,
            ant_type=ara.VpolAntenna,
            require_mc_truth=args.montecarlo
        )>=3,
        "hpol": lambda detector: pol_trigger(
            det=detector,
            ant_type=ara.HpolAntenna,
            require_mc_truth=args.montecarlo
        )>=3,
    }
    script_metadata['detector_description'] = "ARA02 station in relative coordinates"
    script_metadata['trigger_requirement'] = "3/8 tunnel diode triggers in either polarization"

elif args.detector==3:
    detector = ara.ARA03()
    detector.build_antennas(power_threshold=6.46, noisy=not(args.noiseless))
    is_triggered = {
        "global": lambda detector: detector.triggered(
            polarized_antenna_requirement=3,
            require_mc_truth=args.montecarlo,
        ),
        "vpol": lambda detector: pol_trigger(
            det=detector,
            ant_type=ara.VpolAntenna,
            require_mc_truth=args.montecarlo
        )>=3,
        "hpol": lambda detector: pol_trigger(
            det=detector,
            ant_type=ara.HpolAntenna,
            require_mc_truth=args.montecarlo
        )>=3,
    }
    script_metadata['detector_description'] = "ARA03 station in relative coordinates"
    script_metadata['trigger_requirement'] = "3/8 tunnel diode triggers in either polarization"

else:
    raise ValueError("Invalid detector argument")


# Set ice model (just use the default for now)
ice = pyrex.ice

# Set earth model (just use the default for now)
earth = pyrex.earth


# Set neutrino generator with a radius which varies with energy, since higher
# energy events can be seen from further away
if 1e3<=args.energy<1e8:
    radius = 1000
elif 1e8<=args.energy<1e9:
    radius = 2500
elif 1e9<=args.energy<1e10:
    radius = 5000
elif 1e10<=args.energy:
    radius = 10000
else:
    radius = 10000
generator = pyrex.CylindricalGenerator(dr=radius, dz=2800,
                                        energy=args.energy,
                                        shadow=args.shadow,
                                        earth_model=earth,
                                        source="cosmogenic")
script_metadata['generator_size_r'] = generator.dr
script_metadata['generator_size_z'] = generator.dz


# Set secondary interactions
if hasattr(generator, 'interaction_model'):
    generator.interaction_model.include_secondaries = args.secondaries

# Additional metadata
script_metadata['monte_carlo_truth'] = int(args.montecarlo)
try:
    script_metadata['interaction_model_class'] = str(generator.interaction_model)
except AttributeError:
    pass
else:
    script_metadata['secondaries'] = generator.interaction_model.include_secondaries
try:
    script_metadata['flavor_ratio_e'] = generator.ratio[0]
except AttributeError:
    pass
else:
    script_metadata['flavor_ratio_mu'] = generator.ratio[1]
    script_metadata['flavor_ratio_tau'] = generator.ratio[2]

# Write script metadata to output file
outfile.add_analysis_metadata("sim_parameters", script_metadata)

# Set signal model (just use the default for now)
signal_model = pyrex.AskaryanSignal

# Limit oncone angle range to reduce memory consumption. Any signals generated
# within this angle of the Cherenkov cone will simply use the Askaryan signal
# as it appears on the Cherenov cone, since calculations very close to the
# cone consume drastically more memory than those further away.
pyrex.AskaryanSignal.oncone_range = np.radians(1e-3)


# Prepare signal amplitude dataset. This dataset will provide extra information
# which may be useful in characterizing events for analysis, especially if the
# full waveforms are not recorded.
amp_dataset = outfile.create_analysis_dataset(
    "signal_amplitudes", shape=(0, len(detector), 2),
    dtype=np.float_, maxshape=(None, len(detector), 2)
)
amp_dataset.dims[0].label = "events"
amp_dataset.dims[1].label = "antennas"
amp_dataset.dims[2].label = "amplitudes"
amp_dataset.attrs['keys'] = [b"signal", b"signal+noise"]

# Set up the simulation kernel. The cuts described in offcone_max and
# weight_min will speed up the simulation without changing the results much.
# Any signals that would be observed more than 40 degrees away from the
# Cherenkov cone are skipped, since their signal will be too weak to trigger
# the antennas. Any event with a survival weight (probability of traversing its
# path through the Earth) less than 1e-5 will be skipped since it will
# contribute negligibly to the overall effective volume.
kernel = pyrex.EventKernel(
    generator=generator,
    antennas=detector,
    ice_model=ice,
    event_writer=outfile,
    triggers=is_triggered,
    signal_model=signal_model,
    signal_times=np.linspace(-200e-9, 200e-9, 4001),
    offcone_max=40,
    weight_min=(1e-5, 0),
)

# Run the kernel over the specified number of events, with additional code for
# recording the signal amplitudes on each antenna for each event.
for i in range(args.number):
    print(i, end='..', flush=True)
    try:
        detector.clear(reset_noise=True)
    except TypeError:
        for ant in detector:
            ant.clear(reset_noise=True)
    event, triggered = kernel.event()
    row = amp_dataset.shape[0]
    max_waves = 2
    amp_dataset.resize(row+max_waves, axis=0)
    for col, ant in enumerate(detector):
        for j, (sig, wave) in enumerate(zip(ant.signals, ant.all_waveforms)):
            amp_dataset[row+j, col, 0] = np.max(np.abs(sig.values))
            amp_dataset[row+j, col, 1] = np.max(np.abs(wave.values))
    outfile.add_analysis_indices("signal_amplitudes", global_index=i,
                                 start_index=row, length=max_waves)

print()

outfile.add_file_metadata({
    "kernel_completed": datetime.datetime.now().strftime('%Y-%d-%m %H:%M:%S')
})


# Analyze effective volume from this simulation, for the global detector trigger
# as well as any additional triggers defined
with pyrex.File(args.output, 'r') as reader:
    global_triggers = 0
    if 'mc_triggers' in reader:
        other_triggers = np.zeros(reader['mc_triggers'].shape[1])
        other_trigger_keys = list(reader['mc_triggers'].attrs['keys'])
    else:
        other_triggers = []
        other_trigger_keys = []
    for data in reader:
        weight = data.get_particle_info('survival_weight')[0]
        if data.triggered:
            global_triggers += weight
        if other_trigger_keys!=[]:
            for component in data.get_triggered_components():
                i = other_trigger_keys.index(str.encode(component))
                other_triggers[i] += weight

triggers = np.concatenate(([global_triggers], other_triggers))
keys = np.concatenate(([b"global"], other_trigger_keys))

veff = triggers / generator.count * generator.volume
err = np.sqrt(triggers) / generator.count * generator.volume

# Write effective volume data to the output file
with pyrex.File(args.output, 'a') as writer:
    if 'effective_volume' in writer:
        del writer['effective_volume']
    veff_dataset = writer.create_analysis_dataset(
        "effective_volume", data=np.concatenate((np.vstack(veff),
                                                    np.vstack(err)), axis=1)
    )
    veff_dataset.attrs['keys'] = keys
    veff_dataset.attrs['generation_volume'] = generator.volume
    veff_dataset.attrs['total_events_thrown'] = generator.count

outfile.close()