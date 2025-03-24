import numpy as np
import pybullet as p
import open3d as o3d
import antipodal_grasp_detection_helper as helper
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def get_antipodal(pcd):
    """
    Function to compute antipodal grasp given point cloud pcd.
    :param pcd: Point cloud in Open3D format (converted to numpy below).
    :return: Gripper pose (4,) numpy array of gripper pose (x, y, z, theta).
    """
    # Convert pcd to numpy arrays of points and normals
    pc_points = np.asarray(pcd.points)
    pc_normals = np.asarray(pcd.normals)

    # ------------------------------------------------
    # Compute DBSCAN clustering to segment the object
    object_labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=50))
    object_cluster_mask = object_labels == 0  # Assuming first cluster is the target
    cluster_points = pc_points[object_cluster_mask]
    cluster_normals = pc_normals[object_cluster_mask]

    # Cluster based on surface normals
    clustering = DBSCAN(eps=0.1, min_samples=10).fit(cluster_normals)
    plane_labels = clustering.labels_
    valid_indices = plane_labels != -1
    cluster_points = cluster_points[valid_indices]
    cluster_normals = cluster_normals[valid_indices]
    plane_labels = plane_labels[valid_indices]

    # Calculate average normals for each plane
    unique_labels = np.unique(plane_labels)
    average_normals = [np.mean(cluster_normals[plane_labels == label], axis=0) for label in unique_labels]

    # Find antipodal plane pairs
    anti_parallel_threshold = -0.9
    plane_pairs = [
        (i, j)
        for i in range(len(average_normals))
        for j in range(i + 1, len(average_normals))
        if np.dot(average_normals[i], average_normals[j]) < anti_parallel_threshold
    ]

    # Filter pairs based on gripper width
    gripper_width = 0.15
    best_pair = None
    smallest_dist = float('inf')
    for pair in plane_pairs:
        points_1 = cluster_points[plane_labels == unique_labels[pair[0]]]
        points_2 = cluster_points[plane_labels == unique_labels[pair[1]]]
        
        if points_1.size == 0 or points_2.size == 0:
            continue
        dists = np.linalg.norm(points_1[:, None, :] - points_2[None, :, :], axis=-1)
        min_dist = np.min(dists)
        if min_dist < smallest_dist and min_dist <= gripper_width:
            smallest_dist = min_dist
            best_pair = pair

    if not best_pair:
        print(f"Debug: No valid pairs found for gripper width: {gripper_width}")
        print(f"Smallest distance found: {smallest_dist}")
        raise ValueError("No valid plane pair found for the given gripper width.")

    # Compute gripper pose
    plane1_points = cluster_points[plane_labels == unique_labels[best_pair[0]]]
    plane2_points = cluster_points[plane_labels == unique_labels[best_pair[1]]]
    plane1_mean = np.mean(plane1_points, axis=0)
    plane2_mean = np.mean(plane2_points, axis=0)
    midpoint = (plane1_mean + plane2_mean) / 2
    theta = np.arctan2((plane2_mean - plane1_mean)[1], (plane2_mean - plane1_mean)[0])
    gripper_pose = np.array([midpoint[0], midpoint[1], midpoint[2], theta])
    # ------------------------------------------------

    return gripper_pose


def main(n_tries=5):
    # Initialize the world
    world = helper.World()

    # Start grasping loop
    for i in range(n_tries):
        # Get point cloud from cameras in the world
        pcd = world.get_point_cloud()
        # Check point cloud to see if there are still objects to remove
        finish_flag = helper.check_pc(pcd)
        if finish_flag:  # If no more objects -- done!
            print('===============')
            print('Scene cleared')
            print('===============')
            break
        # Visualize the point cloud from the scene
        helper.draw_pc(pcd)
        # Compute antipodal grasp
        gripper_pose = get_antipodal(pcd)
        # Send command to robot to execute
        robot_command = world.grasp(gripper_pose)
        # Robot drops object to the side
        world.drop_in_bin(robot_command)
        # Robot goes to initial configuration and prepares for next grasp
        world.home_arm()

    # Terminate simulation environment once you're done!
    p.disconnect()
    return finish_flag


if __name__ == "__main__":
    flag = main()