from .sawyer_xyz.env_dict import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from .wrappers import DoorOpenNoGripperObs, DoorOpenNoGripperControl, \
    EpisodeLengthWrapper
from .spec import EnvSpec, VecQuantSpec, OBS_SPECS, quants_to_sizes
from .utils import DoorOpenRewardFunctor


def make_env(max_episode_length, seed, use_gripper=True):
    """
    seed: necessarily required bc __init__ is set manually and initializes random_vec
    """
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['door-open-v2-goal-observable'](seed)

    if not use_gripper:
        env = DoorOpenNoGripperObs(env)
        env = DoorOpenNoGripperControl(env)

    assert isinstance(max_episode_length, int) and max_episode_length < 500, \
        "Implementation of MujocoEnv does not allow rollouts with more than 500 steps."
    
    env = EpisodeLengthWrapper(env, max_length=max_episode_length)

    return env


def get_sawyer_env_spec():
    obs_spec = VecQuantSpec.from_desc(OBS_SPECS['default'], quants_to_sizes)
    return EnvSpec(env_name='dooropen',
                   obs_spec=obs_spec,
                   rfunc=DoorOpenRewardFunctor())
