"""
Module containing IREX front-end circuit models.

Contains wrappers for PySpice circuits as well as analytical forms for some
envelope circuits.

"""

import os.path
import numpy as np
from scipy.special import lambertw
from pyrex.signals import Signal

import warnings

import pyrex.custom.pyspice as pyspice

spice_circuits = {}

if pyspice.__available__:
    spice_library_path = os.path.join(os.path.dirname(pyspice.__file__),
                                      'spice_models')
    spice_library = pyspice.SpiceLibrary(spice_library_path)

    # Basic envelope circuit:
    #
    #   Vin---D1>---+---+---out
    #               |   |
    #              C1   R1
    #               |   |
    #               +---+
    #               |
    #              gnd
    #
    basic_envelope_circuit = pyspice.Circuit('Basic Envelope Circuit')
    basic_envelope_circuit.include(spice_library['hsms'])

    basic_envelope_circuit.V('in', 'input', basic_envelope_circuit.gnd,
                             'dc 0 external')
    basic_envelope_circuit.X('D1', 'hsms', 'input', 'output')
    basic_envelope_circuit.C(1, 'output', basic_envelope_circuit.gnd,
                             pyspice.u_pF(20))
    basic_envelope_circuit.R(1, 'output', basic_envelope_circuit.gnd,
                             pyspice.u_Ohm(500))
    
    spice_circuits['basic'] = basic_envelope_circuit

    # Biased envelope circuit:
    #
    #     Vin---C2---+---D1>---+---+---out
    #                |         |   |
    #               R2        C1   R1
    #                |         |   |
    #   Vbias---R3---+         +---+
    #                |         |
    #               D2        gnd
    #                v
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
                       pyspice.u_pF(20))
    biased_envelope_circuit.R(1, 'output', biased_envelope_circuit.gnd,
                       pyspice.u_Ohm(500))

    spice_circuits['biased'] = biased_envelope_circuit

    # Voltage doubler envelope circuit:
    #
    #                       Isrc
    #                        |
    #   Vin---C1---+---D1>---+---C3---out
    #              |         |
    #              ^         |
    #             D2         C2
    #              |         |
    #              +----+----+
    #                   |
    #                  gnd
    #
    doubler_envelope_circuit = pyspice.Circuit('Voltage Doubler Envelope Circuit')
    doubler_envelope_circuit.include(spice_library['hsms'])

    doubler_envelope_circuit.V('in', 'input', doubler_envelope_circuit.gnd,
                               'dc 0 external')
    doubler_envelope_circuit.C(1, 'input', 1,
                               pyspice.u_pF(20))
    doubler_envelope_circuit.X('D1', 'hsms', 1, 'output')
    doubler_envelope_circuit.X('D2', 'hsms', doubler_envelope_circuit.gnd, 1)
    doubler_envelope_circuit.C(2, 'output', doubler_envelope_circuit.gnd,
                               pyspice.u_pF(20))
    # doubler_envelope_circuit.I('src', 2, 3,
    #                            pyspice.u_mA(0.1))
    # doubler_envelope_circuit.V('bias', doubler_envelope_circuit.gnd, 3,
    #                            pyspice.u_V(0.75))
    # doubler_envelope_circuit.C(3, 2, 'output',
    #                            pyspice.u_nF(1))
    doubler_envelope_circuit.R(1, 'output', doubler_envelope_circuit.gnd,
                               pyspice.u_Ohm(500))

    spice_circuits['doubler'] = doubler_envelope_circuit

    # # Log amplifier envelope circuit:
    # #
    # #   Vin---+---C1---+   +-------+-----Vs
    # #         |        |   |       |
    # #         |        8   7   6   5
    # #         |     +--+---+---+---+--+
    # #        R1     |      AD8310     |
    # #         |     +--+---+---+---+--+
    # #         |        1   2   3   4
    # #         |        |   |       |
    # #         +---C2---+  gnd      +-----Vout
    # #         |
    # #        gnd
    # #
    # log_amp_envelope_circuit = pyspice.Circuit('Log Amplifier Envelope Circuit')
    # log_amp_envelope_circuit.include(spice_library['AD8310_MODEL'])

    # log_amp_envelope_circuit.V('in', 'input', log_amp_envelope_circuit.gnd,
    #                            'dc 0 external')
    # log_amp_envelope_circuit.R(1, 'input', log_amp_envelope_circuit.gnd,
    #                            pyspice.u_Ohm(52.3))
    # log_amp_envelope_circuit.C(1, 'input', 'pin8',
    #                            pyspice.u_nF(10))
    # log_amp_envelope_circuit.C(2, log_amp_envelope_circuit.gnd, 'pin1',
    #                            pyspice.u_nF(10))
    # log_amp_envelope_circuit.X('AD8310', 'AD8310_MODEL', 'pin8', 'pin1',
    #                            'pin5/7', 'pin3', 'output', 'pin6', 'pin5/7')
    # log_amp_envelope_circuit.V('s', 'pin5/7', log_amp_envelope_circuit.gnd,
    #                            pyspice.u_V(5))

    # spice_circuits['logamp'] = log_amp_envelope_circuit


    # Bridge rectifier envelope circuit:
    #
    #   +-----------+
    #   |           |
    #   |       +---+---+
    #   |       |       |
    #   |       ^       D1
    #   |      D3       v
    #   |       |       |
    #  Vin      +--gnd  +-----+---+---out
    #   |       |       |     |   |
    #   |      D4       ^
    #   |       v       D2   C1   R1
    #   |       |       |     |   |
    #   |       +---+---+     +---+
    #   |           |         |
    #   +-----------+        gnd
    #
    bridge_envelope_circuit = pyspice.Circuit('Bridge Rectifier Envelope Circuit')
    bridge_envelope_circuit.include(spice_library['hsms'])

    bridge_envelope_circuit.V('in', 'input', 'neg',
                              'dc 0 external')
    bridge_envelope_circuit.X('D1', 'hsms', 'input', 'output')
    bridge_envelope_circuit.X('D2', 'hsms', 'neg', 'output')
    bridge_envelope_circuit.X('D3', 'hsms', bridge_envelope_circuit.gnd, 'input')
    bridge_envelope_circuit.X('D4', 'hsms', bridge_envelope_circuit.gnd, 'neg')
    bridge_envelope_circuit.C(1, 'output', basic_envelope_circuit.gnd,
                              pyspice.u_pF(20))
    bridge_envelope_circuit.R(1, 'output', basic_envelope_circuit.gnd,
                              pyspice.u_Ohm(500))

    spice_circuits['bridge'] = bridge_envelope_circuit



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
def basic_envelope_model(signal, cap=20e-12, res=500):
    """
    Model of a basic diode-capacitor-resistor envelope circuit.

    Passes the input signal through a basic envelope circuit consisting of a
    diode, a capacitor, and a resistor. The diode used is modeled after an
    HSMS 2852 diode.

    Parameters
    ----------
    signal : Signal
        Signal object used as input to the circuit.
    cap : float, optional
        Capacitance (F) of the circuit's capacitor ``C1``.
    res : float, optional
        Resistance (ohm) of the circuit's resistor ``R1``.

    Returns
    -------
    Signal
        Output of the envelope circuit for the given input.

    Notes
    -----
    Ascii depiction of the basic envelope circuit::

        Vin---D1---+---+---out
                   |   |
                  C1   R1
                   |   |
                   +---+
                   |
                  gnd

    """
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

    return Signal(signal.times, v_out, value_type=Signal.Type.voltage)


# Bridge rectifier envelope circuit:
#
#   +-----------+
#   |           |
#   |       +---+---+
#   |       |       |
#   |       ^       D1
#   |      D3       v
#   |       |       |
#  Vin      +--gnd  +-----+---+---out
#   |       |       |     |   |
#   |      D4       ^     |   |
#   |       v       D2   C1   R1
#   |       |       |     |   |
#   |       +---+---+     +---+
#   |           |         |
#   +-----------+        gnd
#
def bridge_rectifier_envelope_model(signal, cap=20e-12, res=500):
    """
    Model of a diode bridge rectifier envelope circuit.

    Passes the input signal through a diode bridge rectifier envelope circuit
    consisting of four diodes in a diode bridge, a capacitor, and a resistor.
    The diode used is modeled after an HSMS 2852 diode.

    Parameters
    ----------
    signal : Signal
        Signal object used as input to the circuit.
    cap : float, optional
        Capacitance (F) of the circuit's capacitor ``C1``.
    res : float, optional
        Resistance (ohm) of the circuit's resistor ``R1``.

    Returns
    -------
    Signal
        Output of the envelope circuit for the given input.

    Notes
    -----
    Ascii depiction of the diode bridge rectifier envelope circuit::

         +-----------+
         |           |
         |       +---+---+
         |       |       |
         |       ^       D1
         |      D3       v
         |       |       |
        Vin      +--gnd  +-----+---+---out
         |       |       |     |   |
         |      D4       ^     |   |
         |       v       D2   C1   R1
         |       |       |     |   |
         |       +---+---+     +---+
         |           |         |
         +-----------+        gnd

    """
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

    for v_in in np.abs(signal.values):
        # Calculate exponent of exponential in lambert function instead of
        # calculating exponential directly to avoid overflows
        a = lambert_exponent + (v_in - v_c)/n/v_t/2
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

    return Signal(signal.times, v_out, value_type=Signal.Type.voltage)
