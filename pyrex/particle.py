"""
Module for particles (neutrinos) and neutrino interactions in the ice.

Included in the module are the Particle class for storing particle/shower
attributes and some Interaction classes which store models describing neutrino
interactions.

"""

from collections.abc import Iterable
from enum import Enum
import inspect
import logging
import os.path
import numpy as np
import scipy.constants
from pyrex.internal_functions import normalize, get_from_enum

logger = logging.getLogger(__name__)


def _read_secondary_data_file(data_directory, flavor, secondary_type,
                              energies=("1e18", "1e18.5", "1e19", "1e19.5",
                                        "1e20", "1e20.5", "1e21")):
    """
    Read a file of secondary inelasticity data.

    Reads from file(s) `data_directory` / `flavor` / dsdy_ `secondary_type`
    _ `energies` [.vec, _tau.vec].

    Parameters
    ----------
    data_directory : str
        Directory containing secondary data.
    flavor : {"muons", "tauon"}
        Flavor of neutrino corresponding to a data sub-directory.
    secondary_type : str
        Type of secondary for which inelasticity data should be returned.
    energies : list of str
        Energies at which the data should be read.

    Returns
    -------
    y_vals : list of ndarray
        Arrays of inelasticity values for each energy in `energies`.
    dsdy_vals : list of ndarray
        Arrays of d(sigma)/dy values corresponding to the `y_vals` for each
        energy in `energies`.

    """
    if flavor=="muons":
        suffix = ".vec"
    elif flavor=="tauon":
        suffix = "_tau.vec"
    else:
        raise ValueError("Unknown flavor value '"+flavor+"'")
    y_vals = []
    dsdy_vals = []
    for energy in energies:
        filename = "dsdy_"+secondary_type+"_"+energy+suffix
        fullname = os.path.join(data_directory, flavor, filename)
        ys = []
        dsdys = []
        with open(fullname) as f:
            for line in f:
                y, dsdy = line.split()
                ys.append(float(y))
                dsdys.append(float(dsdy))
        y_vals.append(np.array(ys))
        dsdy_vals.append(np.array(dsdys))
    return y_vals, dsdy_vals

# Load the probability distributions for secondary data from files
_secondary_data_dir = os.path.join(os.path.dirname(__file__), "data", "secondary")
_y_muon_brems, _dsdy_muon_brems = _read_secondary_data_file(
    _secondary_data_dir, "muons", "brems"
)
_y_muon_epair, _dsdy_muon_epair = _read_secondary_data_file(
    _secondary_data_dir, "muons", "epair"
)
_y_muon_pn, _dsdy_muon_pn = _read_secondary_data_file(
    _secondary_data_dir, "muons", "pn"
)
_y_tauon_brems, _dsdy_tauon_brems = _read_secondary_data_file(
    _secondary_data_dir, "tauon", "brems"
)
_y_tauon_epair, _dsdy_tauon_epair = _read_secondary_data_file(
    _secondary_data_dir, "tauon", "epair"
)
_y_tauon_pn, _dsdy_tauon_pn = _read_secondary_data_file(
    _secondary_data_dir, "tauon", "pn"
)
_y_tauon_hadrdecay, _dsdy_tauon_hadrdecay = _read_secondary_data_file(
    _secondary_data_dir, "tauon", "hadrdecay"
)
_y_tauon_edecay, _dsdy_tauon_edecay = _read_secondary_data_file(
    _secondary_data_dir, "tauon", "edecay"
)
_y_tauon_mudecay, _dsdy_tauon_mudecay = _read_secondary_data_file(
    _secondary_data_dir, "tauon", "mudecay"
)

# Integrate the probability distributions
_int_muon_brems = np.sum(_dsdy_muon_brems, axis=1)
_int_muon_epair = np.sum(_dsdy_muon_epair, axis=1)
_int_muon_pn = np.sum(_dsdy_muon_pn, axis=1)
_int_tauon_brems = np.sum(_dsdy_tauon_brems, axis=1)
_int_tauon_epair = np.sum(_dsdy_tauon_epair, axis=1)
_int_tauon_pn = np.sum(_dsdy_tauon_pn, axis=1)
_int_tauon_hadrdecay = np.sum(_dsdy_tauon_hadrdecay, axis=1)
_int_tauon_edecay = np.sum(_dsdy_tauon_edecay, axis=1)
_int_tauon_mudecay = np.sum(_dsdy_tauon_mudecay, axis=1)

# Calculate the cumulative distributions
_y_cum_muon_brems = (np.cumsum(_dsdy_muon_brems, axis=1)
                     / _int_muon_brems[:,np.newaxis])
_y_cum_muon_epair = (np.cumsum(_dsdy_muon_epair, axis=1)
                     / _int_muon_epair[:,np.newaxis])
_y_cum_muon_pn = (np.cumsum(_dsdy_muon_pn, axis=1)
                  / _int_muon_pn[:,np.newaxis])
_y_cum_tauon_brems = (np.cumsum(_dsdy_tauon_brems, axis=1)
                      / _int_tauon_brems[:,np.newaxis])
_y_cum_tauon_epair = (np.cumsum(_dsdy_tauon_epair, axis=1)
                      / _int_tauon_epair[:,np.newaxis])
_y_cum_tauon_pn = (np.cumsum(_dsdy_tauon_pn, axis=1)
                   / _int_tauon_pn[:,np.newaxis])
_y_cum_tauon_hadrdecay = (np.cumsum(_dsdy_tauon_hadrdecay, axis=1)
                          / _int_tauon_hadrdecay[:,np.newaxis])
_y_cum_tauon_edecay = (np.cumsum(_dsdy_tauon_edecay, axis=1)
                       / _int_tauon_edecay[:,np.newaxis])
_y_cum_tauon_mudecay = (np.cumsum(_dsdy_tauon_mudecay, axis=1)
                        / _int_tauon_mudecay[:,np.newaxis])


class Interaction:
    """
    Base class for describing neutrino interaction attributes.

    Defaults to values which will result in zero probability of interaction.

    Parameters
    ----------
    particle : Particle
        ``Particle`` object for which the interaction is defined.
    kind : optional
        Value of the interaction type. Values should be from the
        ``Interaction.Type`` enum, but integer or string values may work if
        carefully chosen. By default will be chosen by the `choose_interaction`
        method.

    Attributes
    ----------
    particle : Particle
        ``Particle`` object for which the interaction is defined.
    kind : Interaction.Type
        Value of the interaction type.
    inelasticity : float
        Inelasticity value from `choose_inelasticity` distribution for the
        interaction.
    em_frac : float
        Fraction of `particle` energy deposited into an electromagnetic shower.
    had_frac : float
        Fraction of `particle` energy deposited into a hadronic shower.
    total_cross_section
    total_interaction_length
    cross_section
    interaction_length

    See Also
    --------
    Particle : Class for storing particle attributes.

    """

    class Type(Enum):
        """
        Enum containing possible interaction types.

        Attributes
        ----------
        cc, charged_current
        nc, neutral_current
        unknown, undefined

        """
        undefined = 0
        unknown = 0
        charged_current = 1
        cc = 1
        neutral_current = 2
        nc = 2

    def __init__(self, particle, kind=None):
        self.particle = particle
        self.kind = kind
        if self.kind==self.Type.undefined:
            self.kind = self.choose_interaction()
        self.inelasticity = self.choose_inelasticity()
        self.em_frac, self.had_frac = self.choose_shower_fractions()

    @property
    def _metadata(self):
        """Metadata dictionary for writing `Interaction` information."""
        return {
            "name": self.kind.name,
            "kind": self.kind.value,
            "inelasticity": self.inelasticity,
            "em_frac": self.em_frac,
            "had_frac": self.had_frac
        }

    @property
    def kind(self):
        """
        Value of the interaction type.

        Should always be a value from the ``Interaction.Type`` enum. Setting
        with integer or string values may work if carefully chosen.

        """
        return self._interaction_type

    @kind.setter
    def kind(self, int_type):
        if int_type is None:
            self._interaction_type = self.Type.undefined
        else:
            self._interaction_type = get_from_enum(int_type, self.Type)

    def choose_interaction(self):
        """
        Choose an interaction type for the ``particle`` attribute.

        By default, always chooses undefined interaction type.

        Returns
        -------
        Interaction.Type
            Enum value for the interaction type for the particle.

        """
        return self.Type.undefined

    def choose_inelasticity(self):
        """
        Choose an inelasticity for the ``particle`` attribute's shower.

        By default, always returns zeros.

        Returns
        -------
        float
            Inelasticity (y) value for the interaction.

        """
        return 0

    def choose_shower_fractions(self):
        """
        Choose the electromagnetic and hadronic shower fractions.

        By default, always returns zero.

        Returns
        -------
        em_frac : float
            Electromagnetic shower fraction.
        had_frac : float
            Hadronic shower fraction.
        """
        return 0, 0

    @property
    def total_cross_section(self):
        """
        The total neutrino cross section (cm^2) of the ``particle`` type.

        Calculation is determined by whether the ``particle`` is a neutrino
        or antineutrino and is dependent on the energy of the ``particle``.
        Combines the charged-current and neutral-current cross sections. By
        default, always zero so no interaction will occur.

        """
        return 0

    @property
    def cross_section(self):
        """
        The neutrino cross section (cm^2) of the ``particle`` interaction.

        Calculation is determined by whether the ``particle`` is a neutrino
        or antineutrino and what type of interaction it produces, and is
        dependent on the energy of the ``particle``. By default, always zero so
        no interaction will occur.

        """
        return 0

    @property
    def total_interaction_length(self):
        """
        The neutrino interaction length (cmwe) of the ``particle`` type.

        The interaction length is calculated in centimeters of water
        equivalent. Calculation is determined by whether the ``particle`` is a
        neutrino or antineutrino and is dependent on the energy of the
        ``particle``. Combines the charged-current and neutral-current
        interaction lengths.

        """
        return 1 / (scipy.constants.N_A * self.total_cross_section)

    @property
    def interaction_length(self):
        """
        The neutrino interaction length (cmwe) of the ``particle`` interaction.

        The interaction length is calculated in centimeters of water
        equivalent. Calculation is determined by whether the ``particle`` is a
        neutrino or antineutrino and what type of interaction it produces, and
        is dependent on the energy of the ``particle``.

        """
        return 1 / (scipy.constants.N_A * self.cross_section)


class GQRSInteraction(Interaction):
    """
    Class for describing neutrino interaction attributes.

    Calculates values related to the interaction(s) of a given `particle`.
    Values based on GQRS 1998.

    Parameters
    ----------
    particle : Particle
        ``Particle`` object for which the interaction is defined.
    kind : optional
        Value of the interaction type. Values should be from the
        ``Interaction.Type`` enum, but integer or string values may work if
        carefully chosen. By default will be chosen by the `choose_interaction`
        method.

    Attributes
    ----------
    particle : Particle
        ``Particle`` object for which the interaction is defined.
    kind : Interaction.Type
        Value of the interaction type.
    inelasticity : float
        Inelasticity value from `choose_inelasticity` distribution for the
        interaction.
    em_frac : float
        Fraction of `particle` energy deposited into an electromagnetic shower.
    had_frac : float
        Fraction of `particle` energy deposited into a hadronic shower.
    include_secondaries : bool
        If true, secondary interactions will be considered when calculating
        the shower fractions.
    total_cross_section
    total_interaction_length
    cross_section
    interaction_length

    See Also
    --------
    Interaction : Base class for describing neutrino interaction attributes.
    Particle : Class for storing particle attributes.

    Notes
    -----
    Neutrino intractions based on the GQRS Ultrahigh-Energy Neutrino
    Interactions paper [1]_.

    References
    ----------
    .. [1] R. Gandhi et al, "Ultrahigh-Energy Neutrino Interactions."
        Physical Review D **58**, 093009 (1998).
        :doi:`10.1103/PhysRevD.58.093009`

    """
    include_secondaries = True

    def choose_interaction(self):
        """
        Choose an interaction type for the ``particle`` attribute.

        Randomly generates the interaction type (charged-current or
        neutral-current) according to the cross-section ratio.

        Returns
        -------
        Interaction.Type
            Enum value for the interaction type for the particle.

        Notes
        -----
        The interaction type choice is based on the ratio used in AraSim and
        ANITA's icemc. It claims to be based on "[the] Ghandi etal paper,
        updated for the CTEQ6-DIS parton distribution functions (M.H. Reno,
        personal communication)".

        """
        if np.random.rand()<0.6865254:
            return self.Type.charged_current
        else:
            return self.Type.neutral_current

    def choose_inelasticity(self):
        """
        Choose an inelasticity for the ``particle`` attribute's shower.

        Generates an inelasticity based on the interaction type.

        Returns
        -------
        float
            Inelasticity (y) value for the interaction.

        Notes
        -----
        The inelasticity calculation is based on the "old" calculation in
        AraSim (or the pickYGhandietal function in ANITA's icemc). It is
        documented as "A ROUGH PARAMETRIZATION OF PLOT [7] FROM Ghandhi, Reno,
        Quigg, Sarcevic hep-ph/9512364 (the curves are not in their later
        article). There is also a slow energy dependence."

        """
        r_1 = 1 / np.e
        r_2 = 1 - r_1
        rnd = np.random.rand()
        return (-np.log(r_1+rnd*r_2))**2.5

    def choose_shower_fractions(self):
        """
        Choose the electromagnetic and hadronic shower fractions.

        Calculates the maximal electromagnetic and hadronic shower fractions
        based on the primary ``particle`` and randomly generated secondary
        interactions. Method pulled from AraSim, which is unchanged from icemc.

        Returns
        -------
        em_frac : float
            Electromagnetic shower fraction.
        had_frac : float
            Hadronic shower fraction.

        """
        # Calculate shower fractions for primary particle
        if self.kind==self.Type.neutral_current:
            em_frac = 0
            had_frac = self.inelasticity
        elif self.kind==self.Type.charged_current:
            if (self.particle.id==self.particle.Type.electron_neutrino or
                    self.particle.id==self.particle.Type.electron_antineutrino):
                em_frac = 1 - self.inelasticity
                had_frac = self.inelasticity
            elif (self.particle.id==self.particle.Type.muon_neutrino or
                  self.particle.id==self.particle.Type.muon_antineutrino):
                em_frac = 0
                had_frac = self.inelasticity
            elif (self.particle.id==self.particle.Type.tau_neutrino or
                  self.particle.id==self.particle.Type.tau_antineutrino):
                # Treat taus like muons
                em_frac = 0
                had_frac = self.inelasticity
            else:
                raise ValueError("Particle type not supported")
        else:
            raise ValueError("Interaction type not supported")

        # Stop here if no secondary interactions should be considered
        if not self.include_secondaries:
            return em_frac, had_frac

        # Calculate lepton energy for inelasticity distributions
        if self.kind==self.Type.charged_current:
            lepton_energy = self.particle.energy * (1-self.inelasticity)
        elif self.kind==self.Type.neutral_current:
            # No outgoing lepton, therefore no secondaries
            # Just return primary fractions
            return em_frac, had_frac
        else:
            raise ValueError("Secondaries cannot be generated without an "+
                             "interaction type")
        energy_index = int(2*(np.log10(lepton_energy)-18))
        if energy_index<0:
            energy_index = 0
        elif energy_index>6:
            energy_index = 6

        loop_counter = 0
        # Try some reasonable number of times to produce secondaries which
        # conserve energy
        while loop_counter<1000:
            loop_counter += 1
            em_secondaries, had_secondaries = \
                self._choose_secondary_fractions(lepton_energy, energy_index)
            # If the generated secondaries conserve energy, check whether the
            # larger signal will come from primary or secondaries
            if em_secondaries+had_secondaries<=lepton_energy:
                # If the signal from secondaries is larger, return the
                # secondary fractions
                if (em_secondaries+had_secondaries >
                        (em_frac+had_frac)*self.particle.energy):
                    return (em_secondaries / self.particle.energy,
                            had_secondaries / self.particle.energy)
                # Otherwise return the primary fractions
                else:
                    return em_frac, had_frac

    def _choose_secondary_fractions(self, lepton_energy, energy_index):
        """
        Choose electromagnetic and hadronic shower fractions from secondaries.

        Generate random secondary interactions for the given `lepton_energy`
        and return the largest electromagnetic and hadronic fractions from the
        secondaries. Method pulled from AraSim, which is unchanged from icemc.

        Parameters
        ----------
        lepton_energy : float
            The energy (GeV) of the lepton produced in a charged-current
            interaction.
        energy_index : int
            The index relating the `lepton_energy` to the proper set of data.
            Should be calculated as 2*log10(`lepton_energy`)-18.

        Returns
        -------
        em_frac : float
            Maximal electromagnetic shower fraction from secondary
            interactions.
        had_frac : float
            Maximal hadronic shower fraction form secondary interactions.

        """
        em_max = 0
        had_max = 0
        if (self.particle.id==self.particle.Type.muon_neutrino or
                self.particle.id==self.particle.Type.muon_antineutrino):
            # Pick numbers of bremsstrahlung, pair production, and
            # photonuclear interactions based on the integrated
            # secondaries data
            n_brems = np.random.poisson(_int_muon_brems[energy_index])
            n_epair = np.random.poisson(_int_muon_epair[energy_index])
            n_pn = np.random.poisson(_int_muon_pn[energy_index])
            n_tot = n_brems + n_epair + n_pn
            # Generate all secondary interactions and capture the largest
            for _ in range(n_tot):
                rand_interaction = np.random.rand()
                if rand_interaction<n_brems/n_tot:
                    interaction = "brems"
                    cum_dist = _y_cum_muon_brems[energy_index]
                elif rand_interaction<(n_brems+n_epair)/n_tot:
                    interaction = "epair"
                    cum_dist = _y_cum_muon_epair[energy_index]
                else:
                    interaction = "pn"
                    cum_dist = _y_cum_muon_pn[energy_index]
                # Calculate inelasticity according to cumulative
                # distribution
                rand_inelasticity = np.random.rand()
                y = np.interp(rand_inelasticity, cum_dist,
                              np.linspace(0, 1, len(cum_dist)))
                # Store if the largest interaction thus far
                if y*lepton_energy>max(em_max, had_max):
                    if interaction=="brems" or interaction=="epair":
                        em_max = y*lepton_energy
                    elif interaction=="pn":
                        had_max = y*lepton_energy

        elif (self.particle.id==self.particle.Type.tau_neutrino or
              self.particle.id==self.particle.Type.tau_antineutrino):
            # Pick numbers of bremsstrahlung, pair production, and
            # photonuclear interactions based on the integrated
            # secondaries data
            n_brems = np.random.poisson(_int_tauon_brems[energy_index])
            n_epair = np.random.poisson(_int_tauon_epair[energy_index])
            n_pn = np.random.poisson(_int_tauon_pn[energy_index])
            n_tot = n_brems + n_epair + n_pn
            # Generate all secondary interactions and capture the largest
            for _ in range(n_tot):
                rand_interaction = np.random.rand()
                if rand_interaction<n_brems/n_tot:
                    interaction = "brems"
                    cum_dist = _y_cum_tauon_brems[energy_index]
                elif rand_interaction<(n_brems+n_epair)/n_tot:
                    interaction = "epair"
                    cum_dist = _y_cum_tauon_epair[energy_index]
                else:
                    interaction = "pn"
                    cum_dist = _y_cum_tauon_pn[energy_index]
                # Calculate inelasticity according to cumulative
                # distribution
                rand_inelasticity = np.random.rand()
                y = np.interp(rand_inelasticity, cum_dist,
                              np.linspace(0, 1, len(cum_dist)))
                # Store if the largest interaction thus far
                if y*lepton_energy>max(em_max, had_max):
                    if interaction=="brems" or interaction=="epair":
                        em_max = y*lepton_energy
                    elif interaction=="pn":
                        had_max = y*lepton_energy
            # Handle tau decay interaction just like the others
            rand_interaction = np.random.rand()
            if rand_interaction<0.65011:
                interaction = "hadrdecay"
                cum_dist = _y_cum_tauon_hadrdecay[energy_index]
            elif rand_interaction<0.8219:
                interaction = "mudecay"
                cum_dist = _y_cum_tauon_mudecay[energy_index]
            else:
                interaction = "edecay"
                cum_dist = _y_cum_tauon_edecay[energy_index]
            # Calculate inelasticity according to cumulative
            # distribution
            rand_inelasticity = np.random.rand()
            y = np.interp(rand_inelasticity, cum_dist,
                          np.linspace(0, 1, len(cum_dist)))
            # Store if the largest interaction thus far
            if y*lepton_energy>max(em_max, had_max):
                if interaction=="edecay":
                    em_max = y*lepton_energy
                elif interaction=="hadrdecay":
                    had_max = y*lepton_energy

        return em_max, had_max

    @property
    def total_cross_section(self):
        """
        The total neutrino cross section (cm^2) of the ``particle`` type.

        Calculation is determined by whether the ``particle`` is a neutrino
        or antineutrino and is dependent on the energy of the ``particle``.
        Combines the charged-current and neutral-current cross sections.

        """
        # Particle
        if self.particle.id.value>0:
            coeff = 7.84e-36
            power = 0.363
        # Antiparticle
        elif self.particle.id.value<0:
            coeff = 7.80e-36
            power = 0.363
        else:
            raise ValueError("Unable to calculate cross section without a"+
                             " particle type")
        # Calculate cross section based on GQRS 1998
        return coeff * self.particle.energy**power

    @property
    def cross_section(self):
        """
        The neutrino cross section (cm^2) of the ``particle`` interaction.

        Calculation is determined by whether the ``particle`` is a neutrino
        or antineutrino and what type of interaction it produces, and is
        dependent on the energy of the ``particle``.

        """
        # Particle
        if self.particle.id.value>0:
            if self.kind==self.Type.charged_current:
                coeff = 5.53e-36
                power = 0.363
            elif self.kind==self.Type.neutral_current:
                coeff = 2.31e-36
                power = 0.363
            else:
                raise ValueError("Unable to calculate cross section without an"
                                 +" interaction type")
        # Antiparticle
        elif self.particle.id.value<0:
            if self.kind==self.Type.charged_current:
                coeff = 5.52e-36
                power = 0.363
            elif self.kind==self.Type.neutral_current:
                coeff = 2.29e-36
                power = 0.363
            else:
                raise ValueError("Unable to calculate cross section without an"
                                 +" interaction type")
        else:
            raise ValueError("Unable to calculate cross section without a"+
                             " particle type")
        # Calculate cross section based on GQRS 1998
        return coeff * self.particle.energy**power


class CTWInteraction(GQRSInteraction):
    """
    Class for describing neutrino interaction attributes.

    Calculates values related to the interaction(s) of a given `particle`.
    Values based on CTW 2011.

    Parameters
    ----------
    particle : Particle
        ``Particle`` object for which the interaction is defined.
    kind : optional
        Value of the interaction type. Values should be from the
        ``Interaction.Type`` enum, but integer or string values may work if
        carefully chosen. By default will be chosen by the `choose_interaction`
        method.

    Attributes
    ----------
    particle : Particle
        ``Particle`` object for which the interaction is defined.
    kind : Interaction.Type
        Value of the interaction type.
    inelasticity : float
        Inelasticity value from `choose_inelasticity` distribution for the
        interaction.
    em_frac : float
        Fraction of `particle` energy deposited into an electromagnetic shower.
    had_frac : float
        Fraction of `particle` energy deposited into a hadronic shower.
    include_secondaries : bool
        If true, secondary interactions will be considered when calculating
        the shower fractions.
    total_cross_section
    total_interaction_length
    cross_section
    interaction_length

    See Also
    --------
    Interaction : Base class for describing neutrino interaction attributes.
    Particle : Class for storing particle attributes.

    Notes
    -----
    Neutrino intractions based on the CTW High Energy Neutrino-Nucleon Cross
    Sections paper [1]_. Secondary generation method to determine shower
    fractions was pulled from AraSim, which is unchanged from icemc.

    References
    ----------
    .. [1] A. Connolly et al, "Calculation of High Energy Neutrino-Nucleon
        Cross Sections and Uncertainties Using the MSTW Parton Distribution
        Functions and Implications for Future Experiments." Physical Review D
        **83**, 113009 (2011). :arxiv:`1102.0691`
        :doi:`10.1103/PhysRevD.83.113009`

    """
    def choose_interaction(self):
        """
        Choose an interaction type for the ``particle`` attribute.

        Randomly generates the interaction type (charged-current or
        neutral-current) according to the cross-section ratio.

        Returns
        -------
        Interaction.Type
            Enum value for the interaction type for the particle.

        Notes
        -----
        The interaction type choice is based on the ratio equation in the CTW
        2011 paper [1]_ (Equation 8).

        References
        ----------
        .. [1] A. Connolly et al, "Calculation of High Energy Neutrino-Nucleon
            Cross Sections and Uncertainties Using the MSTW Parton Distribution
            Functions and Implications for Future Experiments." Physical Review
            D **83**, 113009 (2011). :arxiv:`1102.0691`
            :doi:`10.1103/PhysRevD.83.113009`

        """
        d_0 = 1.76
        d_1 = 0.252162
        d_2 = 0.0256
        eps = np.log10(self.particle.energy)
        nc_frac = d_1 + d_2 * np.log(eps - d_0)
        if np.random.rand()<nc_frac:
            return self.Type.neutral_current
        else:
            return self.Type.charged_current

    def choose_inelasticity(self):
        """
        Choose an inelasticity for the ``particle`` attribute's shower.

        Generates a random inelasticity from an inelasticity distribution based
        on the interaction type.

        Returns
        -------
        float
            Inelasticity (y) value for the interaction.

        Notes
        -----
        The inelasticity calculation is based on Equations 14-18 in the CTW
        2011 paper [1]_.

        References
        ----------
        .. [1] A. Connolly et al, "Calculation of High Energy Neutrino-Nucleon
            Cross Sections and Uncertainties Using the MSTW Parton Distribution
            Functions and Implications for Future Experiments." Physical Review
            D **83**, 113009 (2011). :arxiv:`1102.0691`
            :doi:`10.1103/PhysRevD.83.113009`

        """
        eps = np.log10(self.particle.energy)
        # Step 1
        is_low_y = bool(np.random.rand() < 0.128*np.sin(-0.197*(eps-21.8)))
        # Step 2
        if is_low_y:
            a_0 = 0
            a_1 = 0.0941
            a_2 = 4.72
            a_3 = 0.456
        else:
            if self.kind==self.Type.charged_current:
                # Particle
                if self.particle.id.value>0:
                    a_0 = -0.008
                    a_1 = 0.26
                    a_2 = 3
                    a_3 = 1.7
                # Antiparticle
                elif self.particle.id.value<0:
                    a_0 = -0.0026
                    a_1 = 0.085
                    a_2 = 4.1
                    a_3 = 1.7
                else:
                    raise ValueError("Unable to calculate inelasticity without"+
                                     " a particle type")
            elif self.kind==self.Type.neutral_current:
                a_0 = -0.005
                a_1 = 0.23
                a_2 = 3
                a_3 = 1.7
            else:
                raise ValueError("Unable to calculate inelasticity without an"
                                 +" interaction type")
        # Equations 16 & 17
        c_1 = a_0 - a_1*np.exp(-(eps-a_2)/a_3)
        c_2 = 2.55 - 0.0949*eps
        # Step 3
        r = np.random.rand()
        if is_low_y:
            y_min = 0
            y_max = 1e-3
            # Equation 14
            return c_1 + (r*(y_max-c_1)**(1-1/c_2) +
                          (1-r)*(y_min-c_1)**(1-1/c_2))**(c_2/(c_2-1))
        else:
            y_min = 1e-3
            y_max = 1
            # Equation 15
            return (y_max-c_1)**r / (y_min-c_1)**(r-1) + c_1

    @property
    def total_cross_section(self):
        """
        The total neutrino cross section (cm^2) of the ``particle`` type.

        Calculation is determined by whether the ``particle`` is a neutrino
        or antineutrino and is dependent on the energy of the ``particle``.
        Combines the charged-current and neutral-current cross sections.
        Based on Equation 7 and Table III of the CTW 2011 paper.

        """
        # Total cross section should be sum of nc and cc cross sections

        # Particle
        if self.particle.id.value>0:
            c_0_cc = -1.826
            c_0_nc = -1.826
            c_1_cc = -17.31
            c_1_nc = -17.31
            c_2_cc = -6.406
            c_2_nc = -6.448
            c_3_cc = 1.431
            c_3_nc = 1.431
            c_4_cc = -17.91
            c_4_nc = -18.61
        # Antiparticle
        elif self.particle.id.value<0:
            c_0_cc = -1.033
            c_0_nc = -1.033
            c_1_cc = -15.95
            c_1_nc = -15.95
            c_2_cc = -7.247
            c_2_nc = -7.296
            c_3_cc = 1.569
            c_3_nc = 1.569
            c_4_cc = -17.72
            c_4_nc = -18.30
        else:
            raise ValueError("Unable to calculate cross section without a"+
                             " particle type")
        # Calculate cross section based on CTW 2011
        eps = np.log10(self.particle.energy)
        log_term_cc = np.log(eps - c_0_cc)
        power_cc = (c_1_cc + c_2_cc*log_term_cc + c_3_cc*log_term_cc**2
                    + c_4_cc/log_term_cc)
        log_term_nc = np.log(eps - c_0_nc)
        power_nc = (c_1_nc + c_2_nc*log_term_nc + c_3_nc*log_term_nc**2
                    + c_4_nc/log_term_nc)
        return 10**power_cc + 10**power_nc

    @property
    def cross_section(self):
        """
        The neutrino cross section (cm^2) of the ``particle`` interaction.

        Calculation is determined by whether the ``particle`` is a neutrino
        or antineutrino and what type of interaction it produces, and is
        dependent on the energy of the ``particle``. Based on Equation 7 and
        Table III of the CTW 2011 paper.

        """
        # Particle
        if self.particle.id.value>0:
            if self.kind==self.Type.charged_current:
                c_0 = -1.826
                c_1 = -17.31
                c_2 = -6.406
                c_3 = 1.431
                c_4 = -17.91
            elif self.kind==self.Type.neutral_current:
                c_0 = -1.826
                c_1 = -17.31
                c_2 = -6.448
                c_3 = 1.431
                c_4 = -18.61
            else:
                raise ValueError("Unable to calculate cross section without an"
                                 +" interaction type")
        # Antiparticle
        elif self.particle.id.value<0:
            if self.kind==self.Type.charged_current:
                c_0 = -1.033
                c_1 = -15.95
                c_2 = -7.247
                c_3 = 1.569
                c_4 = -17.72
            elif self.kind==self.Type.neutral_current:
                c_0 = -1.033
                c_1 = -15.95
                c_2 = -7.296
                c_3 = 1.569
                c_4 = -18.30
            else:
                raise ValueError("Unable to calculate cross section without an"
                                 +" interaction type")
        else:
            raise ValueError("Unable to calculate cross section without a"+
                             " particle type")
        # Calculate cross section based on CTW 2011
        eps = np.log10(self.particle.energy)
        log_term = np.log(eps - c_0)
        power = c_1 + c_2*log_term + c_3*log_term**2 + c_4/log_term
        return 10**power


# Preferred interaction model
NeutrinoInteraction = CTWInteraction



class Particle:
    """
    Class for storing particle attributes.

    Parameters
    ----------
    particle_id
        Identification value of the particle type. Values should be from the
        ``Particle.Type`` enum, but integer or string values may work if
        carefully chosen. ``Particle.Type.undefined`` by default.
    vertex : array_like
        Vector position (m) of the particle.
    direction : array_like
        Vector direction of the particle's velocity.
    energy : float
        Energy (GeV) of the particle.
    interaction_model : optional
        Class to use to describe interactions of the particle. Should inherit
        from (or behave like) the base ``Interaction`` class.
    interaction_type : optional
        Value of the interaction type. Values should be from the
        ``Interaction.Type`` enum, but integer or string values may work if
        carefully chosen. By default, the `interaction_model` will choose an
        interaction type.
    weight : float, optional
        Total Monte Carlo weight of the particle. The calculation of this
        weight depends on the particle generation method, but this value should
        be the total weight representing the probability of this particle's
        event occurring.

    Attributes
    ----------
    id : Particle.Type
        Identification value of the particle type.
    vertex : array_like
        Vector position (m) of the particle.
    direction : array_like
        (Unit) vector direction of the particle's velocity.
    energy : float
        Energy (GeV) of the particle.
    interaction : Interaction
        Instance of the `interaction_model` class to be used for calculations
        related to interactions of the particle.
    weight : float
        Total Monte Carlo weight of the particle. Given by either the value
        specified on initialization or the product of ``survival_weight`` and
        ``interaction_weight``.
    survival_weight : float
        Monte Carlo weight of the particle surviving to its vertex. Represents
        the probability that the particle does not interact along its path
        through the Earth.
    interaction_weight : float
        Monte Carlo weight of the particle interacting at its vertex.
        Represents the probability that the the particle interacts specifically
        at its given vertex.

    See Also
    --------
    Interaction : Base class for describing neutrino interaction attributes.

    """
    class Type(Enum):
        """
        Enum containing possible particle types.

        Values based on the PDG particle numbering scheme.
        http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf

        Attributes
        ----------
        e, e_minus, electron
        e_plus, positron
        nu_e, electron_neutrino
        nu_e_bar, electron_antineutrino
        mu, mu_minus, muon
        mu_plus, antimuon
        nu_mu, muon_neutrino
        nu_mu_bar, muon_antineutrino
        tau, tau_minus, tauon
        tau_plus, antitau
        nu_tau, tau_neutrino
        nu_tau_bar, tau_antineutrino
        unknown, undefined

        """
        undefined = 0
        unknown = 0
        electron = 11
        e = 11
        e_minus = 11
        positron = -11
        e_plus = -11
        electron_neutrino = 12
        nu_e = 12
        electron_antineutrino = -12
        nu_e_bar = -12
        muon = 13
        mu = 13
        mu_minus = 13
        antimuon = -13
        mu_plus = -13
        muon_neutrino = 14
        nu_mu = 14
        muon_antineutrino = -14
        nu_mu_bar = -14
        tau = 15
        tauon = 15
        tau_minus = 15
        antitau = -15
        tau_plus = -15
        tau_neutrino = 16
        nu_tau = 16
        tau_antineutrino = -16
        nu_tau_bar = -16

    def __init__(self, particle_id, vertex, direction, energy,
                 interaction_model=NeutrinoInteraction, interaction_type=None,
                 weight=None):
        self.id = particle_id
        self.vertex = np.array(vertex)
        self.direction = normalize(direction)
        self.energy = energy
        if inspect.isclass(interaction_model):
            self.interaction = interaction_model(self, kind=interaction_type)
        else:
            raise ValueError("Particle class interaction_model must be a class")
        self.survival_weight = None
        self.interaction_weight = None
        self._forced_weight = weight

    @property
    def _metadata(self):
        """Metadata dictionary for writing `Particle` information."""
        meta = {
            "particle_name": self.id.name,
            "particle_id": self.id.value,
            "vertex_x": self.vertex[0],
            "vertex_y": self.vertex[1],
            "vertex_z": self.vertex[2],
            "direction_x": self.direction[0],
            "direction_y": self.direction[1],
            "direction_z": self.direction[2],
            "energy": self.energy,
            "survival_weight": (self.survival_weight
                                if self.survival_weight is not None
                                else 1),
            "interaction_weight": (self.interaction_weight
                                   if self.interaction_weight is not None
                                   else 1),
            "weight": self.weight,
            "interaction_class": str(type(self.interaction)),
        }
        for key, val in self.interaction._metadata.items():
            meta["interaction_"+key] = val
        return meta

    @property
    def id(self):
        """
        Identification value of the particle type.

        Should always be a value from the ``Particle.Type`` enum. Setting with
        integer or string values may work if carefully chosen.

        """
        return self._id

    @id.setter
    def id(self, particle_id):
        if particle_id is None:
            self._id = self.Type.undefined
        else:
            self._id = get_from_enum(particle_id, self.Type)

    @property
    def weight(self):
        """
        Total Monte Carlo weight of the particle

        Given by either the value specified on initialization or the product of
        ``survival_weight`` and ``interaction_weight``.

        """
        if self._forced_weight is not None:
            return self._forced_weight
        weight = 1
        if self.survival_weight is not None:
            weight *= self.survival_weight
        if self.interaction_weight is not None:
            weight *= self.interaction_weight
        return weight



class Event:
    """
    Class for storing a tree of `Particle` objects representing an event.

    The event may be comprised of any number of root `Particle` objects
    specified at initialization. Each `Particle` in the tree may have any
    number of child `Particle` objects. Iterating the tree will return all
    `Particle` objects, but in no guaranteed order.

    Parameters
    ----------
    roots : Particle or list of Particle
        Root `Particle` objects for the event tree.

    Attributes
    ----------
    roots : Particle or list of Particle
        Root `Particle` objects for the event tree.

    See Also
    --------
    Particle : Class for storing particle attributes.

    """
    def __init__(self, roots):
        if isinstance(roots, Iterable):
            self.roots = roots
        else:
            self.roots = [roots]
        for root in self.roots:
            if not isinstance(root, Particle):
                raise ValueError("Root elements must be Particle objects")
        self._all = [particle for particle in self.roots]
        self._children = [[] for _ in range(len(self.roots))]

    @property
    def _metadata(self):
        """List of metadata dictionaries of the `Particle` objects."""
        return [particle._metadata for particle in self]

    def add_children(self, parent, children):
        """
        Add the given `children` to the `parent` `Particle` object.

        Parameters
        ----------
        parent : Particle
            `Particle` object in the tree to act as the parent to the
            `children`.
        children : Particle or list of Particle
            `Particle` objects to be added as children of the `parent`.

        Raises
        ------
        ValueError
            If the `parent` is not a part of the event tree.

        See Also
        --------
        Particle : Class for storing particle attributes.

        """
        if parent not in self._all:
            raise ValueError("Parent particle is not in the event tree")
        else:
            parent_index = self._all.index(parent)
        if not isinstance(children, Iterable):
            children = [children]
        new_index_start = len(self._all)
        self._all.extend(children)
        indices = [new_index_start+i for i in range(len(children))]
        self._children.extend([[] for _ in indices])
        self._children[parent_index].extend(indices)

    def get_children(self, parent):
        """
        Get the children of the given `parent` `Particle` object.

        Parameters
        ----------
        parent : Particle
            `Particle` object in the tree.

        Returns
        -------
        list of Particle
            List of the `Particle` objects which are children of the `parent`.

        Raises
        ------
        ValueError
            If the `parent` is not a part of the event tree.

        See Also
        --------
        Particle : Class for storing particle attributes.

        """
        if parent not in self._all:
            raise ValueError("Parent particle is not in the event tree")
        else:
            parent_index = self._all.index(parent)
        return [self._all[i] for i in self._children[parent_index]]

    def get_parent(self, child):
        """
        Get the parent of the given `child` `Particle` object.

        Parameters
        ----------
        child : Particle or None
            `Particle` object in the tree. ``None`` if the `child` has no
            parent.

        Returns
        -------
        Particle
            `Particle` object which is the parent of the `child`.

        Raises
        ------
        ValueError
            If the `child` is not a part of the event tree.

        See Also
        --------
        Particle : Class for storing particle attributes.

        """
        if child not in self._all:
            raise ValueError("Child particle is not in the event tree")
        else:
            child_index = self._all.index(child)
        for parent_index, child_indices in enumerate(self._children):
            if child_index in child_indices:
                return self._all[parent_index]

    def get_from_level(self, level):
        """
        Get all `Particle` objects some `level` deep into the event tree.

        Parameters
        ----------
        level : int
            Level of the event tree to scan. Root `Particle` objects at level
            zero.

        Returns
        -------
        list of Particle
            All `Particle` objects at the given `level` in the tree.

        See Also
        --------
        Particle : Class for storing particle attributes.

        """
        # This method could be sped up by working exclusively with indices
        # and self._children, only grabbing the corresponding Particle objects
        # from self._all at the very end. But using self.get_children is clear
        # and as long as the trees are relatively small shouldn't be a problem.
        current_level = 0
        particles = self.roots
        while current_level<level:
            previous = particles
            particles = []
            for p in previous:
                particles.extend(self.get_children(p))
            current_level += 1
        return particles

    # Allow direct iteration of the event by traversing self._all
    def __iter__(self):
        yield from self._all

    def __len__(self):
        return len(self._all)
