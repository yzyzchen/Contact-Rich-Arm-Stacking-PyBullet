import random
import numpy as np
from tower_stacking_planner_helper import World
import pdb
np.set_printoptions(precision=3, suppress=True)


GRIPPER_WIDTH = 0.2
BLOCK_WIDTH = 0.05
BLOCK_HEIGHT = 0.1
# DROP_POSE_1 = np.array([0., -0.17, 0.1, 0., 0.05])
DROP_POSE = np.array([0., -0.12, 0.1, 0., 0.05])
STACK_POSE = np.array([0., -0.3, 0.1, 0, 0.05])
ROBOT_HOME = np.array([0.25, 0.25, 0.25, 0., 0.])


def interpolate_poses(initial_pose, final_pose, n):
    """
    Interpolates poses between an initial and final pose.
    :param initial_pose: Starting pose [x, y, z, ang, gripper_width].
    :param final_pose: Target pose [x, y, z].
    :param n: Number of interpolated steps.
    :return: List of interpolated poses as numpy arrays.
    """
    x_initial, y_initial, z_initial, ang, gripper_width = initial_pose
    x_final, y_final, z_final = final_pose
    x_interp = np.linspace(x_initial, x_final, n)
    y_interp = np.linspace(y_initial, y_final, n)
    z_interp = np.linspace(z_initial, z_final, n)
    interpolated_poses = [
        np.array([x, y, z, ang, gripper_width]) for x, y, z in zip(x_interp, y_interp, z_interp)
    ]
    return interpolated_poses


def pick_up(env, obj_idx):
    """
    Perform grasp in simulation for the specified object.
    """
    obj_states = env.get_obj_state()
    obj_pos = obj_states[obj_idx, :3]
    z_offset = 0.15
    pre_grasp_pose = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + z_offset, 0., 0.2])
    env.robot_command([pre_grasp_pose])
    grasp_pose = np.array([obj_pos[0], obj_pos[1], obj_pos[2] - 0.025, 0., 0.2])
    waypoints = interpolate_poses(pre_grasp_pose, grasp_pose[:3], 5)
    env.robot_command(waypoints)
    grasp_pose[-1] = 0.0
    env.robot_command([grasp_pose])
    grasp_pose[2] += z_offset
    env.robot_command([grasp_pose])
    return grasp_pose


def push(env, obj_idx, direction, distance):
    """
    Perform a push action on the specified object.
    """
    obj_states = env.get_obj_state()
    obj_pos = obj_states[obj_idx, :3]
    target_pos = obj_pos + np.array(direction) * distance
    rob_state = env.get_robot_state()
    pre_push_pose_1 = np.array([rob_state[0], rob_state[1], rob_state[2], np.pi / 2, 0.])
    env.robot_command([pre_push_pose_1])
    pre_push_pose_2 = np.array([obj_pos[0], obj_pos[1] + 0.15, obj_pos[2] + 0.3, rob_state[3], 0.])
    env.robot_command([pre_push_pose_2])
    pre_push_pose = np.array([obj_pos[0], obj_pos[1] + 0.15, obj_pos[2], rob_state[3], 0.])
    env.robot_command([pre_push_pose])
    waypoints = interpolate_poses(pre_push_pose, target_pos, 5)
    env.robot_command(waypoints)


def stack():
    """
    Function to stack objects.
    :return: Average height of objects.
    """
    env = World()
    # Initialize object state and drop pose
    obj_state = env.get_obj_state()
    NEW_DROP_POSE = DROP_POSE.copy()

    for obj_idx in range(len(obj_state)):
        print(f"\n--- Processing Object {obj_idx} ---")
        # Push object if it's not the last one
        if obj_idx < len(obj_state) - 1:
            push(env, obj_idx, [0., -1., 0.], 0.2)

            NEW_STACK_POSE = STACK_POSE
            NEW_STACK_POSE[2] += obj_idx * BLOCK_HEIGHT
        
            # Move robot up after push
            rob_state = env.get_robot_state()
            rob_state[2] += 0.2
            env.robot_command([np.array([rob_state[0], rob_state[1], rob_state[2], 0., 0.])])
            print(f"Robot State After Push: {rob_state}")
            # Pick up the object
            pick_up(env, obj_idx)
            # Move to pre-drop pose
            pre_drop_pose = np.array([NEW_DROP_POSE[0], NEW_DROP_POSE[1], NEW_DROP_POSE[2] + 0.05, 0., 0.])
            env.robot_command([pre_drop_pose])
            print(f"Pre-drop pose reached: {pre_drop_pose}")
            # Move to exact drop pose before opening gripper
            env.robot_command([NEW_DROP_POSE])
            print(f"Drop pose reached: {NEW_DROP_POSE}")
            # Open the gripper to release the object
            drop_pose = NEW_DROP_POSE.copy()
            drop_pose[4] = 0.2  # Fully open gripper
            env.robot_command([drop_pose])
            print("Gripper opened to drop the object.")



        pick_up(env, obj_idx)
        # Move to pre-drop pose
        pre_drop_pose = np.array([NEW_STACK_POSE[0], NEW_STACK_POSE[1], NEW_STACK_POSE[2] + 0.05, 0., 0.])
        env.robot_command([pre_drop_pose])
        print(f"Pre-drop pose reached: {pre_drop_pose}")
        # Move to exact drop pose before opening gripper
        env.robot_command([NEW_STACK_POSE])
        print(f"Drop pose reached: {NEW_STACK_POSE}")
        # Open the gripper to release the object
        drop_pose = NEW_STACK_POSE.copy()
        drop_pose[4] = 0.2  # Fully open gripper
        env.robot_command([drop_pose])
        print("Gripper opened to drop the object.")


        # Move robot up after dropping the object
        post_drop_pose = drop_pose.copy()
        post_drop_pose[2] += 0.20
        env.robot_command([post_drop_pose])
        print("Robot moved up after dropping the object.")
        # Update drop pose for the next object
        # NEW_DROP_POSE[2] += BLOCK_HEIGHT
        print(f"Updated drop pose for next object: {NEW_DROP_POSE}")

        # # indicate the final red pose
        # obj_states = env.get_obj_state()
        # obj_pos = obj_states[obj_idx, :3]
        # print(f"Red block pose is : {obj_pos}")


    # Return robot to home position
    env.robot_command([ROBOT_HOME])
    # Calculate average height
    obj_state = env.get_obj_state()
    print(f"Final Object States: {obj_state}")
    avg_height = np.mean(obj_state[:, 2])
    print("Average Object Height: {:4.3f}".format(avg_height))
    return env, avg_height


if __name__ == "__main__":
    np.random.seed(7)
    random.seed(7)
    stack()