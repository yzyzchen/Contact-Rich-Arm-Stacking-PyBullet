import numpy as np
import pybullet as p
import pybullet_data
import open3d as o3d
import time
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))

from robot import Kuka


def ball(test, o, r):
    r_o = np.linalg.norm(test - o)
    if r_o < r:
        return True
    else:
        return False


def distance(x, d):
    if np.abs(x[0]) < d[0] and np.abs(x[1]) < d[1] and np.abs(x[2]) < d[2]:
        return True
    else:
        return False


class World:
    def __init__(self, u_list=None, p_list=None, visualize=True):
        # initialize the simulator and blocks
        if visualize:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF('plane.urdf', useFixedBase=True)
        p.changeDynamics(planeId, -1, lateralFriction=0.99)

        # set camera
        p.resetDebugVisualizerCamera(cameraDistance=1,
                                     cameraYaw=-60,
                                     cameraPitch=-20,
                                     cameraTargetPosition=[0, 0, 0])

        # set gravity
        p.setGravity(0, 0, -10)

        # add the robot
        self.robot = Kuka()

        for sim_step in range(100):
            p.stepSimulation()

        # drop some blocks in scene
        if u_list is None:
            u_list = [[0.5, -0.35, 1.], [0.5, 0.35, 1.]]
        if p_list is None:
            p_list = [np.array([0., 0., 0.2]), np.array([0.1, 0.1, 0.2])]

        for block in range(2):
            # x_b = np.random.randn() / 15.
            # y_b = np.random.randn() / 15.
            # z_b = 0.3
            u = u_list[block]
            if len(u) == 3:
                q = [np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
                     np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
                     np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
                     np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])]
            else:
                q = u
            block_id = p.loadURDF(os.path.join(file_path, "block.urdf"), [0, 0, 0.05])
            p.changeDynamics(block_id, -1, mass=0.8, lateralFriction=0.89)
            p.resetBasePositionAndOrientation(block_id, p_list[block], q)

        for sim_step in range(300):
            p.stepSimulation()

        self.home_pose = np.array([])

    def get_point_cloud(self):
        camera_number = 2
        img_width = 448
        img_height = 448
        camera_pose = [[0, -0.7, 0.7], [0, 0.6, 0.7], [0.7, 0., 0.7]]  # , [-0.6, 0., 0.7]]
        camera_ori = [[1, 1, 1], [-1, -1, 1], [-1, 0, 1]]  # , [1, 0, 1]]
        camera_ori = [[0, 1, 1], [0, -1, 1], [-1, 0, 1]]  # , [1, 0, 1]]
        point_cloud = np.zeros((img_width * img_height * camera_number, 3))

        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=2.1)

        projection_numpy = np.asarray(projectionMatrix).reshape([4, 4], order='F')
        depth_image = []

        pc_list = []
        n_list = []
        for cameras in range(camera_number):
            # set up camera view matrix
            viewMatrix = p.computeViewMatrix(
                cameraEyePosition=camera_pose[cameras],
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=camera_ori[cameras])

            # take image
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                width=img_width,
                height=img_height,
                viewMatrix=viewMatrix,
                projectionMatrix=projectionMatrix)

            depth_image.append(depthImg)

            # define container for point cloud
            pc_numpy = np.zeros((img_width * img_height, 3))

            # project to world frame and recover pc
            viewMatrix = np.asarray(viewMatrix).reshape([4, 4], order='F')
            tran_pix_world = np.linalg.inv(np.matmul(projection_numpy, viewMatrix))

            count = 0
            for h in range(0, img_height):
                for w in range(0, img_width):
                    x = (2 * w - img_width) / img_width
                    y = -(2 * h - img_height) / img_height  # be careful！ deepth and its corresponding position
                    z = 2 * depth_image[cameras][h, w] - 1
                    pixPos = np.asarray([x, y, z, 1])
                    position = np.matmul(tran_pix_world, pixPos)

                    pc_numpy[count, :] = position[:3] / position[3]
                    count += 1

            # now convert to the o3d format
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(pc_numpy)

            pcd_temp.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd_temp.orient_normals_to_align_with_direction(-np.asarray(camera_ori[cameras]))

            # store results in lists (as numpy)
            pc_list.append(pc_numpy)
            n_list.append(np.asarray(pcd_temp.normals))

            # plt.imshow(rgbImg)
            # plt.show()
            # plt.imshow(depthImg)
            # plt.show()
            # o3d.visualization.draw_geometries([pcd_temp])

        # concatenate your point clouds and make one big one
        pc_combined = np.vstack((pc_list[0], pc_list[1]))  # , pc_list[2], pc_list[3]))
        n_combined = np.vstack((n_list[0], n_list[1]))  # , n_list[2]))  #, n_list[3]))

        ki = []
        for i in range(pc_combined.shape[0]):
            # if distance(point_cloud[i], 0.3):
            if pc_combined[i, 2] > 0.005 and distance(pc_combined[i], [0.2, 0.3, 0.3]):
                ki.append(i)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_combined[ki])
        pcd.normals = o3d.utility.Vector3dVector(n_combined[ki])

        # o3d.visualization.draw_geometries([pcd])
        return pcd

    def grasp(self, gp):
        # plan for robot actions
        # pre-grasp w/ offset
        rotation = np.array([[np.cos(gp[3]), -np.sin(gp[3])],
                             [np.sin(gp[3]), np.cos(gp[3])]])
        go = np.matmul(rotation, np.array([0., -0.02]))
        grasp_pose_robot = np.array([gp[0], gp[1], gp[2], 0., 0.75])
        grasp_offset_a = np.array([go[0], go[1], .4, -gp[3], 0.])
        grasp_offset_b = np.array([go[0], go[1], .4 / 1.5, -gp[3], 0.])
        robot_command = grasp_pose_robot + grasp_offset_a
        for t in range(600):
            self.robot.applyAction(robot_command)
            p.stepSimulation()
        time.sleep(0.3)
        # pre-grasp w/o offset
        robot_command = grasp_pose_robot + grasp_offset_b
        for t in range(600):
            self.robot.applyAction(robot_command)
            p.stepSimulation()
        time.sleep(0.3)
        # close fingers
        grasp_pose_robot[-1] = 0.0
        robot_command = grasp_pose_robot + grasp_offset_b
        for t in range(600):
            self.robot.applyAction(robot_command)
            p.stepSimulation()
        time.sleep(0.3)
        # post-grasp lift
        # grasp_pose_robot[2] = 0.4
        robot_command = grasp_pose_robot + grasp_offset_a
        for t in range(600):
            self.robot.applyAction(robot_command)
            p.stepSimulation()
        time.sleep(0.3)

        return robot_command

    def drop_in_bin(self, robot_command):
        # drop into the bin
        # move to above origin
        for t in range(600):
            self.robot.applyAction(robot_command)
            p.stepSimulation()
        time.sleep(0.3)
        # move above bin
        robot_command[1] = 0.4
        robot_command[2] = 0.4
        for t in range(600):
            self.robot.applyAction(robot_command)
            p.stepSimulation()
        time.sleep(0.3)
        # drop object
        robot_command[-1] = 0.6
        for t in range(600):
            self.robot.applyAction(robot_command)
            p.stepSimulation()
        time.sleep(0.3)

    def home_arm(self):
        self.robot.home_arm()
        time.sleep(0.3)


def draw_pc(pcd):
    o3d.visualization.draw_geometries([pcd])


def check_pc(pcd):
    pc_points = np.asarray(pcd.points)
    if pc_points.size == 0:
        return True
    else:
        return False
