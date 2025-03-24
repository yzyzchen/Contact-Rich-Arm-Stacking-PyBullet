import numpy as np
import pybullet as p
import pybullet_data
from robot import Kuka
import time

np.set_printoptions(precision=3, suppress=True)


class World:
    def __init__(self):
        # initialize the simulator and blocks
        self.physicsClient = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF('plane.urdf', useFixedBase=True)
        p.changeDynamics(planeId, -1, lateralFriction=0.99)

        # set camera
        p.resetDebugVisualizerCamera(cameraDistance=1,
                                     cameraYaw=-60,
                                     cameraPitch=-20,
                                     cameraTargetPosition=[0, 0, 0])

        # Experimenting to make deterministic.
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

        # set gravity
        p.setGravity(0, 0, -10)

        # add the robot
        self.robot = Kuka()

        for sim_step in range(100):
            p.stepSimulation()

        # drop the objects in the scene blocks in scene
        self.block_id = []
        # red block
        ori_red = np.array([0., 0., 0., 1.])  # [0.5, 0.35, 1.]
        pos_red = np.array([0.105, 0.23, 0.2])  # np.array([0.1, 0.1, 0.2])
        block_id = p.loadURDF("red_block.urdf", [0, 0, 0.05])
        p.changeDynamics(block_id, -1, mass=0.8, lateralFriction=0.89)
        p.resetBasePositionAndOrientation(block_id, pos_red, ori_red)
        self.block_id.append(block_id)

        # green block
        ori_green = np.array([0., 0., 0., 1.])  # [0.5, 0.35, 1.]
        pos_green = np.array([0.05, 0.23, 0.2])  # np.array([0.1, 0.1, 0.2])
        block_id = p.loadURDF("green_block.urdf", [0, 0, 0.05])
        p.changeDynamics(block_id, -1, mass=0.8, lateralFriction=0.89)
        p.resetBasePositionAndOrientation(block_id, pos_green, ori_green)
        self.block_id.append(block_id)

        # blue block
        ori_blue = np.array([0., 0., 0., 1.])  # [0.5, 0.35, 1.]
        pos_blue = np.array([-0.005, 0.23, 0.2])  # np.array([0.1, 0.1, 0.2])
        block_id = p.loadURDF("blue_block.urdf", [0, 0, 0.05])
        p.changeDynamics(block_id, -1, mass=0.8, lateralFriction=0.89)
        p.resetBasePositionAndOrientation(block_id, pos_blue, ori_blue)
        self.block_id.append(block_id)

        # box block
        ori_box = np.array([0., 0., 0., 1.])  # [0.5, 0.35, 1.]
        pos_box = np.array([-0.06, 0.23, 0.2])  # np.array([0.1, 0.1, 0.2])
        block_id = p.loadURDF("box.urdf", [0, 0, 0.05])
        p.changeDynamics(block_id, -1, mass=0.8, lateralFriction=0.89)
        p.resetBasePositionAndOrientation(block_id, pos_box, ori_box)

        self.block_id.append(block_id)

        # add fixture
        pos_fix = np.array([0., -0.15, 0.05])  # np.array([0.1, 0.1, 0.2])
        fix_id = p.loadURDF("fixture.urdf", pos_fix, useFixedBase=1)

        for sim_step in range(300):
            p.stepSimulation()

    def get_obj_state(self):
        obj_state = np.zeros((len(self.block_id), 7))
        for ind, block in enumerate(self.block_id):
            pos, ori = p.getBasePositionAndOrientation(block)
            obj_state[ind] = np.concatenate((np.asarray(pos), np.asarray(ori)))

        return obj_state

    def get_robot_state(self):
        return self.robot.get_robot_state()

    def robot_command(self, gp_sequence):
        for gp in gp_sequence:
            rotation = np.array([[np.cos(gp[3]), -np.sin(gp[3])],
                                 [np.sin(gp[3]), np.cos(gp[3])]])
            go = np.matmul(rotation, np.array([0., -0.02]))
            g_width = np.clip(gp[4], 0., 0.2)
            grasp_pose_robot = np.array([gp[0], gp[1], gp[2], 0., g_width])
            grasp_offset = np.array([go[0], go[1], 0.285, -gp[3], 0.])
            robot_command = grasp_pose_robot + grasp_offset

            for t in range(600):
                self.robot.applyAction(robot_command)
                p.stepSimulation()
            time.sleep(0.3)

        return 0

    def home_arm(self):
        self.robot.home_arm()
        time.sleep(0.3)
