import abc
import warnings

import numpy as np


def assert_fully_parsed(func):
    """Decorator function to ensure observations are fully parsed

    Args:
        func (Callable): The function to check

    Returns:
        (Callable): The input function, decorated to assert full parsing
    """
    def inner(obs):
        obs_dict = func(obs)
        assert len(obs) == sum(
            [len(i) if isinstance(i, np.ndarray) else 1 for i in obs_dict.values()]
        ), 'Observation not fully parsed'
        return obs_dict
    return inner


def move(from_xyz, to_xyz, p):
    """Computes action components that help move from 1 position to another

    Args:
        from_xyz (np.ndarray): The coordinates to move from (usually current position)
        to_xyz (np.ndarray): The coordinates to move to
        p (float): constant to scale response

    Returns:
        (np.ndarray): Response that will decrease abs(to_xyz - from_xyz)

    """
    error = to_xyz - from_xyz
    response = p * error

    if np.any(np.absolute(response) > 1.):
        warnings.warn('Constant(s) may be too high. Environments clip response to [-1, 1]')

    return response


class Policy(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def _parse_obs(obs):
        """Pulls pertinent information out of observation and places in a dict.

        Args:
            obs (np.ndarray): Observation which conforms to env.observation_space

        Returns:
            dict: Dictionary which contains information from the observation
        """
        pass

    @abc.abstractmethod
    def get_action(self, obs):
        """Gets an action in response to an observation.

        Args:
            obs (np.ndarray): Observation which conforms to env.observation_space

        Returns:
            np.ndarray: Array (usually 4 elements) representing the action to take
        """
        pass


class Action:
    """
    Represents an action to be taken in an environment.

    Once initialized, fields can be assigned as if the action
    is a dictionary. Once filled, the corresponding array is
    available as an instance variable.
    """
    def __init__(self, structure):
        """
        Args:
            structure (dict): Map from field names to output array indices
        """
        self._structure = structure
        self.array = np.zeros(len(self), dtype='float')

    def __len__(self):
        return sum([1 if isinstance(idx, int) else len(idx) for idx in self._structure.items()])

    def __getitem__(self, key):
        assert key in self._structure, 'This action\'s structure does not contain %s' % key
        return self.array[self._structure[key]]

    def __setitem__(self, key, value):
        assert key in self._structure, 'This action\'s structure does not contain %s' % key
        self.array[self._structure[key]] = value
        
        
class SawyerDoorOpenV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'door_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(
            o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = -1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_door = o_d['door_pos'].copy()
        pos_door[0] -= 0.05
        
        is_near_handle = np.linalg.norm(pos_curr[:2] - pos_door[:2]) < 0.12
        is_near_handle_lvl = np.linalg.norm(pos_curr[2] - pos_door[2]) < 0.04

        # align end effector's Z axis with door handle's Z axis
        if not is_near_handle:
            # print('aligning')
            return pos_door + np.array([0.06, 0.02, 0.2])
        
        # drop down on front edge of door handle
        if not is_near_handle_lvl:
            # print('dropping down')
            return pos_door + np.array([0.06, 0.02, 0.])
        
        return pos_door


class DoorOpenVecPolicy(SawyerDoorOpenV2Policy):
    def __call__(self, obs):
        return self.get_action(obs[0])[np.newaxis]
