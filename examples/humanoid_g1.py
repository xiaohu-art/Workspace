from pathlib import Path
import time
import numpy as np

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_g1" / "scene.xml"

def _joint_nq(jtype: int) -> int:
    if jtype == mujoco.mjtJoint.mjJNT_FREE:  # 7
        return 7
    if jtype == mujoco.mjtJoint.mjJNT_BALL:  # 4
        return 4
    if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return 1
    raise ValueError(f"Unknown joint type: {jtype}")

def build_qstar_with_joint_targets(model, qpos, targets):
    """
    build q_star from q_current and targets
    targets: dict[str, float | sequence]
        - for hinge/slide: scalar
        - for ball: quaternion [w,x,y,z] (normalized)
        - for free: [x,y,z | w,x,y,z]
    """
    q_star = qpos.copy()
    for name, value in targets.items():
        j = model.joint(name)
        nq = _joint_nq(int(j.type))
        start = j.qposadr[0]
        if nq == 1:     # hinge/slide
            v = float(value)
            lo, hi = float(j.range[0]), float(j.range[1])
            v = np.clip(v, lo, hi) # clip to range
            q_star[start] = v
        elif nq == 4:  # ball
            arr = np.asarray(value, dtype=float).reshape(nq)
            norm = np.linalg.norm(arr)
            assert norm == 1.0, "ball joint must be normalized"
            q_star[start:start+nq] = arr
        elif nq == 7:  # free
            arr = np.asarray(value, dtype=float).reshape(nq)
            q_star[start:start+nq] = arr
        else:
            raise ValueError(f"Unknown joint type: {j.type}")
        
    return q_star

def check_joint_limits(model, data, tolerance=1e-4):
    """
    Check if any joints are at or beyond their limits
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        tolerance: Small tolerance for limit detection
    """
    
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name is None:
            continue
            
        joint = model.joint(joint_name)
        joint_type = int(joint.type)
        
        # Only check joints with limits (hinge and slide joints)
        if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            qpos_addr = joint.qposadr[0]
            current_pos = data.qpos[qpos_addr]
            
            # Get joint limits
            joint_range = joint.range
            if np.any(joint_range != 0):  # Joint has limits defined
                lower_limit = float(joint_range[0]) - tolerance
                upper_limit = float(joint_range[1]) + tolerance
                
                # Check for violations
                assert current_pos >= lower_limit and current_pos <= upper_limit, f"Joint {joint_name} is at {current_pos} which is outside the limits {lower_limit} to {upper_limit}"
    return

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    tasks = [
        pelvis_pose_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=5.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=5.0),
        com_task := mink.ComTask(cost=10.0),
        left_foot_task := mink.FrameTask(
            frame_name="left_foot",
            frame_type="site",
            position_cost=100.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        ),
        right_foot_task := mink.FrameTask(
            frame_name="right_foot",
            frame_type="site",
            position_cost=100.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        ),
    ]

    limits = [
        mink.ConfigurationLimit(model)
    ]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("stand")
        posture_task.set_target_from_configuration(configuration)
        pelvis_pose_task.set_target_from_configuration(configuration)
        left_foot_task.set_target_from_configuration(configuration)
        right_foot_task.set_target_from_configuration(configuration)

        root_height_range = np.arange(0.2, 0.76, 0.01)[::-1]
        height_index = 0

        root_pitch_range = np.arange(0.0, 1.57, 0.01)
        pitch_index = 0

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets
            # com_task.set_target(np.array([0, 0, root_height_range[height_index]]))
            com_task.set_target(np.array([0, 0, 0.4]))
            pelvis_pose_task.set_target(
                mink.SE3.from_rotation(
                    rotation=mink.SO3.from_rpy_radians(0, root_pitch_range[pitch_index], 0),
                )
            )

            q_star = build_qstar_with_joint_targets(
                            model, 
                            data.qpos, 
                            {   
                                "left_shoulder_pitch_joint": 0.0,
                                "left_shoulder_roll_joint": 0.0,
                                "right_shoulder_pitch_joint": 0.0,
                                "right_shoulder_roll_joint": 0.0,
                                "left_elbow_joint": 0.0,
                                "right_elbow_joint": 0.0,
                                "waist_pitch_joint": 0.0,
                                "waist_roll_joint": 0.0,
                            }
            )
            data.qpos[:] = q_star
            mujoco.mj_forward(model, data)
            posture_task.set_target(q_star)

            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            check_joint_limits(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            mujoco.mj_camlight(model, data)
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            # height_index += 1
            # if height_index >= len(root_height_range):
            #     input("Press Enter to continue...")
            pitch_index += 1
            if pitch_index >= len(root_pitch_range):
                input("Press Enter to continue...")
            time.sleep(0.2)
