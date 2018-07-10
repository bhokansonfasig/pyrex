"""
Module for particles (neutrinos) and neutrino interactions in the ice.

Included in the module are the Particle class for storing particle/shower
attributes and some Interaction classes which store models describing neutrino
interactions.

"""

from enum import Enum
import inspect
import logging
import numpy as np
from pyrex.internal_functions import normalize, get_from_enum

logger = logging.getLogger(__name__)

AVOGADRO_NUMBER = 6.02e23


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
    total_cross_section
    total_interaction_length
    cross_section
    interaction_length

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
        unknown = 0
        undefined = 0
        cc = 1
        charged_current = 1
        nc = 2
        neutral_current = 2

    def __init__(self, particle, kind=None):
        self.particle = particle
        self.kind = kind
        if self.kind==self.Type.undefined:
            self.kind = self.choose_interaction()
        self.inelasticity = self.choose_inelasticity()

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

        By default, always returns zero.

        Returns
        -------
        float
            Inelasticity (y) value for the interaction.

        """
        return 0

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
        return 1 / (AVOGADRO_NUMBER * self.total_cross_section)

    @property
    def interaction_length(self):
        """
        The neutrino interaction length (cmwe) of the ``particle`` interaction.

        The interaction length is calculated in centimeters of water
        equivalent. Calculation is determined by whether the ``particle`` is a
        neutrino or antineutrino and what type of interaction it produces, and
        is dependent on the energy of the ``particle``.

        """
        return 1 / (AVOGADRO_NUMBER * self.cross_section)


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
    total_cross_section
    total_interaction_length
    cross_section
    interaction_length

    Notes
    -----
    Neutrino intractions based on the GQRS Ultrahigh-Energy Neutrino
    Interactions paper [1]_.

    References
    ----------
    .. [1] R. Gandhi et al, "Ultrahigh-Energy Neutrino Interactions."
        Physical Review D **58**, 093009 (1998).

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


class CTWInteraction(Interaction):
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
    total_cross_section
    total_interaction_length
    cross_section
    interaction_length

    Notes
    -----
    Neutrino intractions based on the CTW High Energy Neutrino-Nucleon Cross
    Sections paper [1]_.

    References
    ----------
    .. [1] A. Connolly et al, "Calculation of High Energy Neutrino-Nucleon
        Cross Sections and Uncertainties Using the MSTW Parton Distribution
        Functions and Implications for Future Experiments." Physical Review D
        **83**, 113009 (2011).

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
            D **83**, 113009 (2011).

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
            D **83**, 113009 (2011).

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
            return c_1 + (r*(y_max-c_1)**(-1/(c_2+1)) +
                          (1-r)*(y_min-c_1)**(-1/(c_2+1)))**(c_2/(c_2-1))
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
        # Based on the form of equation 7 of CTW 2011, the nc and cc cross
        # sections can be summed as long as c_0 is the same for nc and cc.
        # Then c_1, c_2, c_3, and c_4 will be sums of the constants for each
        # current type, while c_0 will stay the same.

        # Particle
        if self.particle.id.value>0:
            c_0 = -1.826
            c_1 = -34.62  # = -17.31 + -17.31
            c_2 = -12.854  # = -6.448 + -6.406
            c_3 = 2.862  # = 1.431 + 1.431
            c_4 = -36.52  # = -18.61 + -17.91
        # Antiparticle
        elif self.particle.id.value<0:
            c_0 = -1.033
            c_1 = -31.9  # = -15.95 + -15.95
            c_2 = -14.543  # = -7.296 + -7.247
            c_3 = 3.138  # = 1.569 + 1.569
            c_4 = -36.02  # = -18.30 + -17.72
        else:
            raise ValueError("Unable to calculate cross section without a"+
                             " particle type")
        # Calculate cross section based on CTW 2011
        eps = np.log10(self.particle.energy)
        log_term = np.log(eps - c_0)
        power = c_1 + c_2*log_term + c_3*log_term**2 + c_4/log_term
        return 10**power

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
        Monte Carlo weight of the particle. The calculation of this weight
        depends on the particle generation method, but this value should be the
        total weight representing the probability of this particle's event
        occurring.

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
        Monte Carlo weight of the particle.

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
        unknown = 0
        undefined = 0
        e = 11
        e_minus = 11
        electron = 11
        e_plus = -11
        positron = -11
        nu_e = 12
        electron_neutrino = 12
        nu_e_bar = -12
        electron_antineutrino = -12
        mu = 13
        mu_minus = 13
        muon = 13
        mu_plus = -13
        antimuon = -13
        nu_mu = 14
        muon_neutrino = 14
        nu_mu_bar = -14
        muon_antineutrino = -14
        tau = 15
        tau_minus = 15
        tauon = 15
        tau_plus = -15
        antitau = -15
        nu_tau = 16
        tau_neutrino = 16
        nu_tau_bar = -16
        tau_antineutrino = -16

    def __init__(self, particle_id, vertex, direction, energy,
                 interaction_model=NeutrinoInteraction, interaction_type=None,
                 weight=1):
        self.id = particle_id
        self.vertex = np.array(vertex)
        self.direction = normalize(direction)
        self.energy = energy
        if inspect.isclass(interaction_model):
            self.interaction = interaction_model(self, kind=interaction_type)
        else:
            raise ValueError("Particle class interaction_model must be a class")
        self.weight = weight

    def __str__(self):
        string = self.__class__.__name__+"("
        for key, val in self.__dict__.items():
            string += key+"="+repr(val)+", "
        return string[:-2]+")"

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

    """
    def __init__(self, roots):
        if isinstance(roots, Particle):
            self.roots = [roots]
        else:
            self.roots = roots
        if len(self.roots)>0:
            if not isinstance(self.roots[0], Particle):
                raise ValueError("Root elements must be Particle objects")
        self._all = [particle for particle in self.roots]
        self._children = [[] for _ in range(len(self.roots))]

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

        """
        if parent not in self._all:
            raise ValueError("Parent particle is not in the event tree")
        else:
            parent_index = self._all.index(parent)
        if isinstance(children, Particle):
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

        Raises
        ------
        ValueError
            If the `parent` is not a part of the event tree.

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

        Raises
        ------
        ValueError
            If the `child` is not a part of the event tree.

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
