from collections import OrderedDict
import re

import numpy as np


from .sawyer_door_v2 import SawyerDoorEnvV2


ALL_V2_ENVIRONMENTS = OrderedDict((
    ('door-open-v2', SawyerDoorEnvV2),
))


def create_hidden_goal_envs():
    hidden_goal_envs = {}
    for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
        d = {}

        def initialize(env, seed=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()
            env._partially_observable = True
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                np.random.set_state(st0)

        d['__init__'] = initialize
        hg_env_name = re.sub("(^|[-])\s*([a-zA-Z])",
                             lambda p: p.group(0).upper(), env_name)
        hg_env_name = hg_env_name.replace("-", "")
        hg_env_key = '{}-goal-hidden'.format(env_name)
        hg_env_name = '{}GoalHidden'.format(hg_env_name)
        HiddenGoalEnvCls = type(hg_env_name, (env_cls, ), d)
        hidden_goal_envs[hg_env_key] = HiddenGoalEnvCls

    return OrderedDict(hidden_goal_envs)


def create_observable_goal_envs():
    observable_goal_envs = {}
    for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
        d = {}

        def initialize(env, seed=None):
            super(type(env), env).__init__()
            
            if seed is not None:
                # should be called after __init__() and before reset()
                env.seed(seed)
                
            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.reset()
            env._freeze_rand_vec = True

        d['__init__'] = initialize
        og_env_name = re.sub("(^|[-])\s*([a-zA-Z])",
                             lambda p: p.group(0).upper(), env_name)
        og_env_name = og_env_name.replace("-", "")

        og_env_key = '{}-goal-observable'.format(env_name)
        og_env_name = '{}GoalObservable'.format(og_env_name)
        ObservableGoalEnvCls = type(og_env_name, (env_cls, ), d)
        observable_goal_envs[og_env_key] = ObservableGoalEnvCls

    return OrderedDict(observable_goal_envs)


ALL_V2_ENVIRONMENTS_GOAL_HIDDEN = create_hidden_goal_envs()
ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE = create_observable_goal_envs()
