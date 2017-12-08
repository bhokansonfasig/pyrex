"""Module containing IREX front-end circuit models"""

import os.path
import numpy as np
from scipy.special import lambertw
from pyrex.signals import Signal


import pyrex.custom.pyspice as pyspice

spice_circuits = {}

if pyspice.__available__:
    spice_library_path = os.path.join(os.path.dirname(pyspice.__file__),
                                      'spice_models')
    spice_library = pyspice.SpiceLibrary(spice_library_path)

    # Basic envelope circuit:
    #
    #   Vin---D1---+---+---out
    #              |   |
    #             C1   R1
    #              |   |
    #              +---+
    #              |
    #             gnd
    #
    basic_envelope_circuit = pyspice.Circuit('Basic Envelope Circuit')
    basic_envelope_circuit.include(spice_library['hsms'])

    basic_envelope_circuit.V('in', 'input', basic_envelope_circuit.gnd,
                             'dc 0 external')
    basic_envelope_circuit.X('D1', 'hsms', 'input', 'output')
    basic_envelope_circuit.C(1, 'output', basic_envelope_circuit.gnd,
                             pyspice.u_pF(220))
    basic_envelope_circuit.R(1, 'output', basic_envelope_circuit.gnd,
                             pyspice.u_Ohm(50))
    
    spice_circuits['basic'] = basic_envelope_circuit

    # Double-diode biased envelope circuit:
    #
    #     Vin---C2---+---D1---+---+---out
    #                |        |   |
    #               R2       C1   R1
    #                |        |   |
    #   Vbias---R3---+        +---+
    #                |        |
    #               D2       gnd
    #                |
    #               gnd
    #
    biased_envelope_circuit = pyspice.Circuit('Biased Envelope Circuit')
    biased_envelope_circuit.include(spice_library['hsms'])

    biased_envelope_circuit.V('in', 'input', biased_envelope_circuit.gnd,
                       'dc 0 external')
    # biasing portion
    biased_envelope_circuit.C(2, 'input', 1,
                       pyspice.u_nF(10))
    biased_envelope_circuit.R(2, 1, 2,
                       pyspice.u_kOhm(1))
    biased_envelope_circuit.X('D2', 'hsms', 2, biased_envelope_circuit.gnd)
    biased_envelope_circuit.R(3, 2, 'bias',
                       pyspice.u_kOhm(1))
    biased_envelope_circuit.V('bias', 'bias', biased_envelope_circuit.gnd,
                       pyspice.u_V(5))
    # envelope portion
    biased_envelope_circuit.X('D1', 'hsms', 1, 'output')
    biased_envelope_circuit.C(1, 'output', biased_envelope_circuit.gnd,
                       pyspice.u_pF(220))
    biased_envelope_circuit.R(1, 'output', biased_envelope_circuit.gnd,
                       pyspice.u_Ohm(50))

    spice_circuits['biased'] = biased_envelope_circuit


# Basic envelope circuit:
#
#   Vin---D1---+---+---out
#              |   |
#             C1   R1
#              |   |
#              +---+
#              |
#             gnd
#
def basic_envelope_model(signal, cap=220e-12, res=50):
    """Model of a basic diode-capacitor-resistor envelope circuit. Takes a
    signal object as the input voltage and returns the output voltage signal
    object."""
    v_c = 0
    v_out = []

    r_d = 25
    i_s = 3e-6
    n = 1.06
    v_t = 26e-3

    # Terms which can be calculated ahead of time to save time in the loop
    charge_exp = np.exp(-signal.dt/(res*cap))
    discharge = i_s*res*(1-charge_exp)
    lambert_factor = n*v_t*res/r_d*(1-charge_exp)
    frac = i_s*r_d/n/v_t
    lambert_exponent = np.log(frac) + frac

    for v_in in signal.values:
        # Calculate exponent of exponential in lambert function instead of
        # calculating exponential directly to avoid overflows
        a = lambert_exponent + (v_in - v_c)/n/v_t
        if a>100:
            # If exponential in lambert function is large enough,
            # use approximation of lambert function
            # (doesn't save time, but does prevent overflows)
            b = np.log(a)
            lambert_term = a - b + b/a
        else:
            # Otherwise, use the lambert function directly
            lambert_term = np.real(lambertw(np.exp(a)))
            if np.isnan(lambert_term):
                # Only seems to happen when np.exp(a) is very close to zero
                # (so lambert_term will also be very close to zero)
                lambert_term = 0

        # Calculate voltage across capacitor after time dt
        v_c = v_c*charge_exp - discharge + lambert_factor*lambert_term
        v_out.append(v_c)

    return Signal(signal.times, v_out, value_type=Signal.ValueTypes.voltage)
