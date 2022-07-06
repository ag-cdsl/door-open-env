import numpy as np

from .mujoco_utils import add_subtree_as_marker
from .spec import Q
from .sawyer_xyz.sawyer_door_v2 import SawyerDoorEnvV2
from .sawyer_xyz import reward_utils


class DoorOpenRewardFunctor:
    def __init__(self):
        self.door_to_ihp_vec = np.array([0.245, -0.12])
        self.door_to_anchor_vec = np.array([-2.5e-1, 0.])
        self.anchor = None  # a point at the axis of door rotation
        self.ihp = None  # initial handle pos
        self.rot_radius = None
    
    def __call__(self, next_obs: np.ndarray, action: np.ndarray):
        """
        action.ndim == next_obs.ndim == 1
        """
        assert next_obs.ndim == action.ndim == 2 \
            and next_obs.shape[0] == action.shape[0] == 1
        next_obs = next_obs.squeeze()
        action = action.squeeze()
        
        theta = self.compute_door_angle(next_obs)

        reward_grab = SawyerDoorEnvV2._reward_grab_effort(action)
        reward_steps = SawyerDoorEnvV2._reward_pos(next_obs, theta)

        reward = sum((
            2.0 * reward_utils.hamacher_product(reward_steps[0], reward_grab),
            8.0 * reward_steps[1],
        ))

        # Override reward on success flag
        target_pos = next_obs[-3:]
        if abs(next_obs[4] - target_pos[0]) <= 0.08:
            reward = 10.0

        return reward

    def compute_door_angle(self, obs):
        chp = obs[4:6]
        a = self.rot_radius
        b = self.rot_radius
        c = np.linalg.norm(chp - self.ihp)
        angle = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
        return -1.0 * angle

    def set_door_pos(self, pos: np.ndarray):
        """door pos is randomized, pos is frozen after env is created"""
        assert pos.ndim == 1, pos.shape[0] == 3
        pos = pos[:2]
        self.ihp = pos + self.door_to_ihp_vec
        self.anchor = pos + self.door_to_anchor_vec
        self.rot_radius = np.linalg.norm(self.ihp - self.anchor)


def render_state(state_spec, x, env, is_offscreen=True):
    if is_offscreen and env.sim._render_context_offscreen is None:
        env.render(offscreen=True)
    if not is_offscreen and env.viewer is None:
        env.render(offscreen=False)

    # eef
    def add_gt_eef():
        body_name = 'hand'
        body_id = env.model.body_name2id(body_name)
        body_pos = env.data.body_xpos[body_id].copy()
        body_quat = env.data.body_xquat[body_id]
        body_pos[0] += 0.3
        # body_quat = np.array([1., 0., 0., 0.])
        qpos = {"r_close": 0,
                "l_close": 0}
        qpos = {
            k: env.data.qpos[env.model.jnt_qposadr[env.model.joint_name2id(k)]] for k in qpos}
        add_subtree_as_marker(env.sim,
                              None if is_offscreen else env.viewer,
                              pos=body_pos,
                              quat=body_quat,
                              root_body_name='right_hand',
                              target_body_name=body_name,
                              joint_qpos=qpos)

    eef_pos = x[state_spec[Q.eef_pos]]
    eef_quat = np.array(
        [0.70577818, -0.00103659, 0.70841842, 0.00440824])
    qpos = {"r_close": 0,
            "l_close": 0}
    #eef_pos[0] += 0.30
    add_subtree_as_marker(env.sim,
                          None if is_offscreen else env.viewer,
                          pos=np.asarray(eef_pos),
                          quat=np.asarray(eef_quat),
                          target_body_name='hand',
                          root_body_name='right_hand',
                          joint_qpos=qpos)

    # handle
    def add_gt_handle():
        from envs.utils.mujoco_utils import Rotation
        handle_geom_name = 'handle'
        handle_geom_id = env.model.geom_name2id(handle_geom_name)
        handle_pos = env.data.geom_xpos[handle_geom_id].copy()
        handle_quat = Rotation.from_matrix(
            env.data.get_geom_xmat(handle_geom_name)).as_quat()

        # handle_body_name = 'door_handle_body'  # geom 'handle'
        # handle_id = env.model.body_name2id(handle_body_name)
        # handle_pos = env.data.body_xpos[handle_id].copy()
        # handle_quat = env.data.body_xquat[handle_id]

        handle_pos[0] += 0.3

        add_subtree_as_marker(env.sim,
                              None if is_offscreen else env.viewer,
                              pos=handle_pos,
                              quat=handle_quat,
                              target_body_name=None,  # handle_body_name
                              target_geom_name=handle_geom_name,
                              root_body_name=None,
                              joint_qpos=None)
    # add_gt_handle()

    # env uses scalar-last native Rotation to produce quat
    handle_quat = x[state_spec[Q.handle_quat]][[3, 0, 1, 2]]
    handle_pos = x[state_spec[Q.handle_pos]]
    # handle_pos[0] += 0.3
    add_subtree_as_marker(env.sim,
                          None if is_offscreen else env.viewer,
                          pos=np.asarray(handle_pos),
                          quat=np.asarray(handle_quat),
                          target_body_name=None,
                          target_geom_name='handle',
                          root_body_name='door_handle_body',  # None
                          joint_qpos=None)

    # render
    image = env.render(offscreen=is_offscreen)
    if is_offscreen:
        env.sim._render_context_offscreen._markers[:] = []

    return image
