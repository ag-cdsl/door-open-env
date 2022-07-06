from collections import defaultdict, deque

import numpy as np
import scipy.spatial
import mujoco_py


def reverse_mapping(arr):
    res = defaultdict(list)
    for x, y in enumerate(arr):
        res[y].append(x)
    return res


def bfs_traversal(root, tree_map):
    q = deque([root])
    while q:
        curr = q.popleft()
        for child in tree_map[curr]:
            q.append(child)
        yield curr


class Rotation(scipy.spatial.transform.Rotation):
    """
    Provides scalar-first quat interface
    """
    @classmethod
    def from_quat(cls, quat):
        return super().from_quat(quat[..., [1, 2, 3, 0]])

    def as_quat(self):
        return super().as_quat()[..., [3, 0, 1, 2]]


def compute_vgeom_params(geom_id: int,
                         parent_pos: np.ndarray,
                         parent_rot: Rotation,
                         sim: mujoco_py.MjSim):
    """
    parent_pos: global cartesian coords
    parent_rot: rot wrt worldbody
    """
    geom_xpos = parent_pos + parent_rot.apply(sim.model.geom_pos[geom_id])
    geom_xrot = parent_rot * Rotation.from_quat(sim.model.geom_quat[geom_id])
    geom_xmat = geom_xrot.as_matrix()

    geom_type = int(sim.model.geom_type[geom_id])
    if geom_type == 7:  # mesh
        # why *2: https://roboti.us/forum/index.php?threads/visualizing-a-mesh.4070/
        dataid = 2 * int(sim.model.geom_dataid[geom_id])
        size = None
    else:  # primitive
        dataid = -1
        size = sim.model.geom_size[geom_id]
        if geom_type == 5:
            size = np.array([size[0], size[0], size[1]])

    rgba = np.array([0.5, 0.5, 0.5, 0.9])
    
    # label = str(sim.model.geom_id2name(geom_id))
    label = ""

    params = {
        "type": geom_type,
        "dataid": dataid,
        "pos": geom_xpos,
        "mat": geom_xmat,
        "rgba": rgba,
        "label": label
    }
    if size is not None:
        params["size"] = size
    return params


def add_subtree_as_marker(sim: mujoco_py.MjSim,
                          viewer: mujoco_py.MjViewer,
                          pos,
                          quat,
                          target_body_name=None,
                          target_geom_name=None,
                          root_body_name=None,
                          joint_qpos: dict = None):
    """
    Adds markers for all geoms in subtree of body with name`root_body_name`
    
    pos, quat are for the body with name `target_body_name`
        or for the geom with name `target_geom_name`
    
    quat: scalar-first
    """
    if viewer is not None:
        add_marker = viewer.add_marker
    else:
        add_marker = sim._render_context_offscreen.add_marker
    
    if target_geom_name is not None:
        assert target_body_name is None, "only one target should be specified"
        target_geom_id = sim.model.geom_name2id(target_geom_name)
        target_body_id = sim.model.geom_bodyid[target_geom_id]
        
        # override pos and quat to ref body instead of geom
        geom_xrot = Rotation.from_quat(quat)
        geom_xpos = pos      
        body_rot = geom_xrot * Rotation.from_quat(sim.model.geom_quat[target_geom_id]).inv()
        body_pos = geom_xpos - body_rot.apply(sim.model.geom_pos[target_geom_id])
        quat = body_rot.as_quat()
        pos = body_pos
    else:
        assert target_body_name is not None, "at least one target should be specified"
        target_body_id = sim.model.body_name2id(target_body_name)
    
    body2children = reverse_mapping(sim.model.body_parentid)
    body2geoms = reverse_mapping(sim.model.geom_bodyid)
    body2posrot = {target_body_id: (pos, Rotation.from_quat(quat))}
    
    for body_id in bfs_traversal(target_body_id, body2children):
        if body_id == target_body_id:
            body_pos, body_rot = body2posrot[body_id]
        else:
            parent_id = sim.model.body_parentid[body_id]
            child_pos, child_rot = body2posrot[parent_id]
            
            body_pos = child_pos + child_rot.apply(sim.model.body_pos[body_id])
            
            # account for slide joints
            if joint_qpos is not None and sim.model.body_jntnum[body_id] == 1:
                joint_idx = sim.model.body_jntadr[body_id]
                if sim.model.jnt_type[joint_idx] == 2:  # slide joint
                    axis = sim.model.jnt_axis[joint_idx]
                    body_pos = body_pos + axis * joint_qpos[sim.model.joint_id2name(joint_idx)]
            
            body_rot = child_rot * Rotation.from_quat(sim.model.body_quat[body_id])
            body2posrot[body_id] = (body_pos, body_rot)
            
        for geom_id in body2geoms[body_id]:
            params = compute_vgeom_params(geom_id, body_pos, body_rot, sim)
            add_marker(**params)
    
    # traverse up to root
    if root_body_name is None:
        root_body_id = target_body_id
    else:
        root_body_id = sim.model.body_name2id(root_body_name)
    body_id = target_body_id
    while body_id != root_body_id:
        child_id = body_id
        body_id = sim.model.body_parentid[body_id]
        child_pos, child_rot = body2posrot[child_id]
        body_rot = child_rot * Rotation.from_quat(sim.model.body_quat[child_id]).inv()
        body_pos = child_pos - body_rot.apply(sim.model.body_pos[child_id])
        body2posrot[body_id] = (body_pos, body_rot)
        for geom_id in body2geoms[body_id]:
            params = compute_vgeom_params(geom_id, body_pos, body_rot, sim)
            add_marker(**params)
