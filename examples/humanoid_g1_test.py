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

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=5.0,
            orientation_cost=5.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=5.0),
        left_foot_task := mink.FrameTask(
            frame_name="left_foot",
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        right_foot_task := mink.FrameTask(
            frame_name="right_foot",
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    collision_pairs = [
        (["left_ankle_roll_collision", "right_ankle_roll_collision"], ["floor"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.05,
    )

    limits = [
        mink.ConfigurationLimit(model),
        collision_avoidance_limit,
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
        pelvis_orientation_task.set_target_from_configuration(configuration)

        root_height_range = np.arange(0.1, 0.76, 0.01)
        root_height_range = root_height_range[::-1]
        height_index = 0

        root_pitch_range = np.arange(0.0, 1.57, 0.01)
        pitch_index = 0

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets.
            pelvis_orientation_task.set_target(
                mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3.from_rpy_radians(0, root_pitch_range[pitch_index], 0),
                    translation=np.array([0, 0, root_height_range[height_index]]),
                )
            )

            q_star = build_qstar_with_joint_targets(model, 
                                                    data.qpos, 
                                                    {
                                                        "waist_pitch_joint": 0.0,
                                                        "waist_roll_joint": 0.0,
                                                    }
                                                    )
            data.qpos[:] = q_star
            mujoco.mj_forward(model, data)
            posture_task.set_target(q_star)

            # left foot target: [-0.00142, 0.11851, 0.0383]
            # right foot target: [-0.00142, -0.11851, 0.0383]
            left_foot_task.set_target(
                mink.SE3.from_translation(
                    translation=np.array([-0.00142, 0.11851, 0.0383]),
                )
            )
            right_foot_task.set_target(
                mink.SE3.from_translation(
                translation=np.array([-0.00142, -0.11851, 0.0383]),
            ))

            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            height_index += 1
            if height_index >= len(root_height_range):
                input("Press Enter to continue...")
            # pitch_index += 1
            # if pitch_index >= len(root_pitch_range):
            #     input("Press Enter to continue...")
            time.sleep(0.2)
