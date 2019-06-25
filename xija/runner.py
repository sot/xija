# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Efficiently run multiple xija models on a list of OR's.
"""

from chandra_aca.transform import calc_aca_from_targ


DEFAULT_MODELS = ('aca', 'dpa', 'dea', 'acisfp', 'tcylaft6')  # etc


class ModelComp:
    """
    Descriptor to specify a model component
    """
    def __init__(self, limit=None, model_err=None):
        self.limit = limit
        self.model_err = model_err

    def __get__(self, instance, owner_cls):
        if instance is None:
            # When called without an instance, return self to allow access
            # to descriptor attributes.
            return self
        else:
            return getattr(instance, self.name)

    def __set__(self, instance, value):
        setattr(instance, self.name, value)

    def __set_name__(self, owner_cls, name):
        self.name = name

        if not hasattr(owner_cls, 'comp_names'):
            owner_cls.comp_names = []
        owner_cls.comp_names.append(name)


class ModelRunner:
    """
    Class object that knows how to run a particular model.  This
    needs to be subclassed with specifics of the model components
    that needs to be set and relevant model values that are output.
    """
    def run(self, obs_req, start):
        """
        Run the model for the specified ``obs_req``

        This sets an attribute ``final_state``
        """
        self.states = self.make_states(obs_req)

        # Now make the model, initialize the states, run, and set ``final_state``

    def make_states(self, obs_req, start):
        """
        Make state values required to run the model for ``obs_req``.

        Includes:

        - Generate a maneuver to the new attitude
        - Assume a SIM translation (if necessary) in the middle of the maneuver
          (is there a better choice?)
        - Potentially replicate some of the ORviewer logic for going to zero-FEPs.
          If this is not done then the answers for ACIS models are conservative.
        """
        pass

    def check_against_limits(self):
        """
        Check that the computed temperature(s) are within specified limits.

        This could in theory be a tri-state
        value: OK, not OK, and almost OK (meaning that the worst case violation
        was within e.g. 20% of the model error).

        :returns: bool (or str?)
        """
        pass


class AcaModelRunner(ModelRunner):
    """
    Specific model components required to run the ACA model.

    Note the limit and model_err values are defaults and could be set
    directly (from ORviewer) with something like
    ``AcaModelRunner.aacccdpt.limit = -7.9``.
    """
    # These class attributes represent the initial state and required data
    # values to run the model.

    pitch = ModelComp(float)
    eclipse = ModelComp(bool)
    aacccdpt = ModelComp(float, limit=-8.5, model_err=1.0)  # Node
    aca0 = ModelComp(float)  # Pseudo-node


class ObsReqRunner:
    def __init__(self, ephemeris, eclipse, model_names=DEFAULT_MODELS):
        """
        :param ephemeris: dict of ephemeris {time (cxcsec), x, y, z}
        """
        self.ephemeris = ephemeris  # needs validation and processing
        self.obs_reqs = {}
        self.models = []

        self.model_names = model_names.copy()

        for model_name in model_names:
            model_runner_name = model_name.capitalize() + 'ModelRunner'
            self.models[model_name] = globals()[model_runner_name]()

    def add_obs_req(self, obsid, att_targ, offset_targ_y, offset_targ_z,
                    off_nom_roll,  # ??
                    detector, n_fep, sim_z, duration):
        """
        Add an OR observation to the set of available observations.

        This is a one-time operation before the first call to ``run()``.  This
        also does the conversion of ``att
        """
        att_aca = calc_aca_from_targ(att_targ, offset_targ_y, offset_targ_z)

        self.obs_reqs[obsid] = {'att_aca': att_aca,
                                'att_targ': att_targ}
        # etc

    def set_initial_state(self, comp_names, comp_values):
        """Supply initial state values for every component name in the
        set of models in the ObsReqRunner object.

        This represents the complete state at the end of the currently planned
        schedule, effectively the continuity.

        Examples of names that need to be provided are: pitch, aoattqt<N>, ccd_count,
        fep_count, aaccdpt (node value), 1cbat, sim_px (pseudo-node values), sim_z.

        This does assume that comp_names are unique, i.e. there are no models
        that use the same component name for two different values.  That is
        currently OK, and should not be a problem in the future.

        IMPLEMENTATION NOTE: can comp_values from MATLAB be a list including
        different data types?  If not then this method needs to operate on
        one comp_name/value pair at once.  This is conceptually nicer but
        probably a bit slower.

        :param comp_names: list of component names
        :param comp_values: list of values corresponding to comp_names

        """
        pass

    def run(self, obsids, start):
        """
        Run the available models for the list of ``obsids``.

        This assumes the continuity state then does the following for each obsid:

        - Generate a maneuver to the new attitude

        - Assume a SIM translation (if necessary) in the middle of the maneuver
          (is there a better choice?)

        - Potentially replicate some of the ORviewer logic for going to zero-FEPs.
          If this is not done then the answers for ACIS modesl are conservative.

        - Determine states for each model component

        - Compute each of the models and place relevant results in the corresponding
          ModelRunner object, which is accessible via the self.models dict.

        - Call the ``check_against_limits`` method for each model for the obsid.
          The logical-or of this is returned.  This could in theory be a tri-state
          value: OK, not OK, and almost OK (meaning that the worst case violation
          was within e.g. 20% of the model error).

        - self.obs_reqs[obsid][model_name]['states'] will be a dict of the input
          states supplied to ``set_data`` for each model component for each
          model.

        - self.obs_reqs[obsid][model_name]['final_state'] will be a dict of the
          final model values for each model component for deeper inspection if
          necessary.  In particular the node and pseudo-node ending temperatures
          would be accessible.

        :param obsids: list of obsids
        :param start: start time (any DateTime-compatible format)

        :returns: list of bool corresponding to thermal check is OK for each obsid.
        """

        all_checks_ok_list = []

        for obsid in obsids:
            all_checks_ok = True

            for model_name in self.model_names:
                model = self.models[model_name]
                model.run(self.obs_reqs[obsid], start)
                model_check_ok = model.checks_against_limits()

                self.obs_reqs[obsid][model_name] = {
                    'final_state': model.final_state,
                    'states': model.states,
                    'check_ok': model_check_ok}

                all_checks_ok &= model_check_ok

            all_checks_ok_list.append(all_checks_ok)

        return all_checks_ok
