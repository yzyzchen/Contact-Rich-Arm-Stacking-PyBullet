import numpy as np
import time
import os
from scipy.spatial import KDTree
import random
try:
    import open3d as o3d
    visualize = True
except ImportError:
    print('To visualize you need to install Open3D. \n \t>> You can use "$ pip install open3d"')
    visualize = False

from assignment_5_helper import ICPVisualizer, load_point_cloud, view_point_cloud, quaternion_matrix, \
    quaternion_from_axis_angle, load_pcs_and_camera_poses, save_point_cloud


def transform_point_cloud(point_cloud, t, R):
    """
    Transform a point cloud applying a rotation and a translation
    :param point_cloud: np.arrays of size (N, 6)
    :param t: np.array of size (3,) representing a translation.
    :param R: np.array of size (3,3) representing a 3D rotation matrix.
    :return: np.array of size (N,6) resulting in applying the transformation (t,R) on the point cloud point_cloud.
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    translated_points = np.dot(point_cloud[:, :3], R.T) + t
    transformed_point_cloud = np.hstack((translated_points, point_cloud[:, 3:]))  # TODO: Replace None with your result
    # ------------------------------------------------
    return transformed_point_cloud


def merge_point_clouds(point_clouds, camera_poses):
    """
    Register multiple point clouds into a common reference and merge them into a unique point cloud.
    :param point_clouds: List of np.arrays of size (N_i, 6)
    :param camera_poses: List of tuples (t_i, R_i) representing the camera i pose.
              - t: np.array of size (3,) representing a translation.
              - R: np.array of size (3,3) representing a 3D rotation matrix.
    :return: np.array of size (N, 6) where $$N = sum_{i=1}^K N_i$$
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    transformed_point_clouds = []

    for i, cloud in enumerate(point_clouds):
        t, R = camera_poses[i]
        transformed_point_cloud = transform_point_cloud(cloud, t, R)
        transformed_point_clouds.append(transformed_point_cloud)

    merged_point_cloud = np.vstack(transformed_point_clouds)   # TODO: Replace None with your result
    # ------------------------------------------------
    return merged_point_cloud


def find_closest_points(point_cloud_A, point_cloud_B):
    """
    Find the closest point in point_cloud_B for each element in point_cloud_A.
    :param point_cloud_A: np.array of size (n_a, 6)
    :param point_cloud_B: np.array of size (n_b, 6)
    :return: np.array of size(n_a,) containing the closest point indexes in point_cloud_B
            for each point in point_cloud_A
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    kdtree_B = KDTree(point_cloud_B[:, :3])
    _, closest_points_indxs = kdtree_B.query(point_cloud_A[:, :3])
    # ------------------------------------------------
    return closest_points_indxs


def find_best_transform(point_cloud_A, point_cloud_B):
    """
    Find the transformation 2 corresponded point clouds.
    Note 1: We assume that each point in the point_cloud_A is corresponded to the point in point_cloud_B at the same location.
        i.e. point_cloud_A[i] is corresponded to point_cloud_B[i] forall 0<=i<N
    :param point_cloud_A: np.array of size (N, 6) (scene)
    :param point_cloud_B: np.array of size (N, 6) (model)
    :return:
         - t: np.array of size (3,) representing a translation between point_cloud_A and point_cloud_B
         - R: np.array of size (3,3) representing a 3D rotation between point_cloud_A and point_cloud_B
    Note 2: We transform the model to match the scene.
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    t = None    # TODO: Replace None with your result
    R = None    # TODO: Replace None with your result
    # ------------------------------------------------
    pmean, qmean = np.mean(point_cloud_A[:, :3], axis=0), np.mean(point_cloud_B[:, :3], axis=0)
    W = (point_cloud_A[:, :3] - pmean).T @ (point_cloud_B[:, :3] - qmean)
    U, _, VT = np.linalg.svd(W)
    R = U @ VT
    t = pmean - R @ qmean
    return t, R


def icp_step(point_cloud_A, point_cloud_B, t_init, R_init):
    """
    Perform an ICP iteration to find a new estimate of the pose of the model point cloud with respect to the scene pointcloud.
    :param point_cloud_A: np.array of size (N_a, 6) (scene)
    :param point_cloud_B: np.array of size (N_b, 6) (model)
    :param t_init: np.array of size (3,) representing the initial transformation candidate
                    * It may be the output from the previous iteration
    :param R_init: np.array of size (3,3) representing the initial rotation candidate
                    * It may be the output from the previous iteration
    :return:
        - t: np.array of size (3,) representing a translation estimate between point_cloud_A and point_cloud_B
        - R: np.array of size (3,3) representing a 3D rotation estimate between point_cloud_A and point_cloud_B
        - correspondences: np.array of size(n_a,) containing the closest point indexes in point_cloud_B
            for each point in point_cloud_A
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    t = None    # TODO: Replace None with your result
    R = None    # TODO: Replace None with your result
    point_cloud_B_transformed = transform_point_cloud(point_cloud_B, t_init, R_init)
    correspondences = find_closest_points(point_cloud_A, point_cloud_B_transformed)  # TODO: Replace None with your result
    correspondences_mask = np.array(correspondences)
    t, R = find_best_transform(point_cloud_A, point_cloud_B[correspondences_mask])
    # ------------------------------------------------
    return t, R, correspondences


def icp(point_cloud_A, point_cloud_B, num_iterations=50, t_init=None, R_init=None, visualize=True):
    """
    Find the
    :param point_cloud_A: np.array of size (N_a, 6) (scene)
    :param point_cloud_B: np.array of size (N_b, 6) (model)
    :param num_iterations: <int> number of icp iteration to be performed
    :param t_init: np.array of size (3,) representing the initial transformation candidate
    :param R_init: np.array of size (3,3) representing the initial rotation candidate
    :param visualize: <bool> Whether to visualize the result
    :return:
         - t: np.array of size (3,) representing a translation estimate between point_cloud_A and point_cloud_B
         - R: np.array of size (3,3) representing a 3D rotation estimate between point_cloud_A and point_cloud_B
    """
    if t_init is None:
        t_init = np.zeros(3)
    if R_init is None:
        R_init = np.eye(3)
    if visualize:
        vis = ICPVisualizer(point_cloud_A, point_cloud_B)
    t = t_init
    R = R_init
    correspondences = None  # Initialization waiting for a value to be assigned
    if visualize:
        vis.view_icp(R=R, t=t)
    for i in range(num_iterations):
        # ------------------------------------------------
        # FILL WITH YOUR CODE

        t, R, correspondences = icp_step(point_cloud_A, point_cloud_B, t, R)
        # ------------------------------------------------
        if visualize:
            vis.plot_correspondences(correspondences)   # Visualize point correspondences
            time.sleep(.5)  # Wait so we can visualize the correspondences
            vis.view_icp(R, t)  # Visualize icp iteration

    return t, R


def filter_point_cloud(point_cloud):
    """
    Remove unnecessary point given the scene point_cloud.
    :param point_cloud: np.array of size (N,6)
    :return: np.array of size (n,6) where n <= N
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    x_mask = np.logical_and(point_cloud[:, 0] >= -0.45, point_cloud[:, 0] <= 0.45)
    y_mask = np.logical_and(point_cloud[:, 1] >= -0.45, point_cloud[:, 1] <= 0.45)
    z_mask = np.logical_and(point_cloud[:, 2] >= 0.01, point_cloud[:, 2] <= np.inf)
    logical_mask = np.logical_and.reduce([x_mask, y_mask, z_mask])

    def rgb_to_hsv(arr):
        arr = np.asarray(arr, dtype=np.float32)
        arr_max, arr_min = arr.max(-1), arr.min(-1)
        delta = arr_max - arr_min
        h = np.zeros_like(arr_max)
        s = np.where(arr_max > 0, delta / arr_max, 0)
        v = arr_max

        idx = delta > 0
        red_max = (arr[..., 0] == arr_max) & idx
        green_max = (arr[..., 1] == arr_max) & idx
        blue_max = (arr[..., 2] == arr_max) & idx

        h[red_max] = (arr[red_max, 1] - arr[red_max, 2]) / delta[red_max]
        h[green_max] = 2.0 + (arr[green_max, 2] - arr[green_max, 0]) / delta[green_max]
        h[blue_max] = 4.0 + (arr[blue_max, 0] - arr[blue_max, 1]) / delta[blue_max]

        h = (h / 6.0) % 1.0
        return np.stack([h, s, v], axis=-1)

    hsv = rgb_to_hsv(point_cloud[:, 3:])
    color_mask = np.logical_and(hsv[:, 0] > 0.125, hsv[:, 0] < 0.25)
    mask = np.logical_and(logical_mask, color_mask)
    filtered_pc = point_cloud[mask]
    # ------------------------------------------------
    return filtered_pc

def ransac_icp_step(point_cloud_A, point_cloud_B, t_init, R_init, ransac_iters=10, ransac_thresh=0.1):
    best_inliers = []
    best_t = np.zeros(3)
    best_R = np.eye(3)

    point_cloud_B_transformed = transform_point_cloud(point_cloud_B, t_init, R_init)

    for _ in range(ransac_iters):
        sampled_indices = random.sample(range(len(point_cloud_A)), k=min(3, len(point_cloud_A)))
        sampled_correspondences = np.array(sampled_indices)

        kdtree_B = KDTree(point_cloud_B_transformed[sampled_correspondences][:, :3])
        distances, correspondences = kdtree_B.query(point_cloud_A[sampled_correspondences][:, :3])

        t_ransac, R_ransac = find_best_transform(
            point_cloud_A[np.array(correspondences)],
            point_cloud_B[np.array(correspondences)]
        )

        transformed_B = transform_point_cloud(point_cloud_B, t_ransac, R_ransac)
        distances = np.linalg.norm(point_cloud_A[:, :3] - transformed_B[:, :3], axis=1)
        inliers = np.where(distances < ransac_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_t = t_ransac
            best_R = R_ransac

    t, R = find_best_transform(point_cloud_A[best_inliers], point_cloud_B[best_inliers])
    correspondences = best_inliers

    return t, R, correspondences


def custom_icp(point_cloud_A, point_cloud_B, num_iterations=50, t_init=None, R_init=None, visualize=True):
    """
        Find the
        :param point_cloud_A: np.array of size (N_a, 6) (scene)
        :param point_cloud_B: np.array of size (N_b, 6) (model)
        :param num_iterations: <int> number of icp iteration to be performed
        :param t_init: np.array of size (3,) representing the initial transformation candidate
        :param R_init: np.array of size (3,3) representing the initial rotation candidate
        :param visualize: <bool> Whether to visualize the result
        :return:
             - t: np.array of size (3,) representing a translation estimate between point_cloud_A and point_cloud_B
             - R: np.array of size (3,3) representing a 3D rotation estimate between point_cloud_A and point_cloud_B
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE (OPTIONAL)
    t = t_init if t_init is not None else np.zeros(3)
    R = R_init if R_init is not None else np.eye(3)


    if visualize:
        vis = ICPVisualizer(point_cloud_A, point_cloud_B)
        vis.view_icp(R=R, t=t)

    for i in range(num_iterations):
        t, R, correspondences = icp_step(point_cloud_A, point_cloud_B, t, R)

        if visualize:
            vis.plot_correspondences(correspondences)
            time.sleep(0.5)
            vis.view_icp(R, t)

    # t, R = icp(point_cloud_A, point_cloud_B, num_iterations=num_iterations, t_init=t_init, R_init=R_init, visualize=visualize)  #TODO: Edit as needed (optional)
    # ------------------------------------------------
    return t, R



# ===========================================================================

# Test functions:

def transform_point_cloud_example(path_to_pointcloud_files, visualize=True):
    pc_source = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med.ply'))  # Source
    pc_goal = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med_tr.ply'))  # Transformed Goal
    t_gth = np.array([-0.5, 0.5, -0.2])
    r_angle = np.pi / 3
    R_gth = quaternion_matrix(np.array([np.cos(r_angle / 2), 0, np.sin(r_angle / 2), 0]))
    pc_tr = transform_point_cloud(pc_source, t=t_gth, R=R_gth)  # Apply your transformation to the source point cloud
    # Paint the transformed in red
    pc_tr[:, 3:] = np.array([.73, .21, .1]) * np.ones((pc_tr.shape[0], 3))  # Paint it red
    if visualize:
        # Visualize first without transformation
        print('Printing the source and goal point clouds')
        view_point_cloud([pc_source, pc_goal])
        # Visualize the transformation
        print('Printing the transformed output (in red) along source and goal point clouds')
        view_point_cloud([pc_source, pc_goal, pc_tr])
    else:
        # Save the pc so we can visualize them using other software
        save_point_cloud(np.concatenate([pc_source, pc_goal], axis=0), 'tr_pc_example_no_transformation',
                     path_to_pointcloud_files)
        save_point_cloud(np.concatenate([pc_source, pc_goal, pc_tr], axis=0), 'tr_pc_example_transform_applied',
                     path_to_pointcloud_files)
        print('Transformed point clouds saved as we cannot visualize them.\n Use software such as Meshlab to visualize them.')


def reconstruct_scene(path_to_pointcloud_files, visualize=True):
    pcs, camera_poses = load_pcs_and_camera_poses(path_to_pointcloud_files)
    pc_reconstructed = merge_point_clouds(pcs, camera_poses)
    if visualize:
        print('Displaying reconstructed point cloud scene.')
        view_point_cloud(pc_reconstructed)
    else:
        print(
            'Reconstructed scene point clouds saved as we cannot visualize it.\n Use software such as Meshlab to visualize them.')
        save_point_cloud(pc_reconstructed, 'reconstructed_scene_pc', path_to_pointcloud_files)


def perfect_model_icp(path_to_pointcloud_files, visualize=True):
    # Load the model
    pcB = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med.ply'))  # Model
    pcB[:, 3:] = np.array([.73, .21, .1]) * np.ones((pcB.shape[0], 3))  # Paint it red
    pcA = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med.ply'))  # Perfect scene
    # Apply transfomation to scene so they differ
    t_gth = np.array([0.4, -0.2, 0.2])
    r_angle = np.pi / 2
    R_gth = quaternion_matrix(np.array([np.cos(r_angle / 2), 0, np.sin(r_angle / 2), 0]))
    pcA = transform_point_cloud(pcA, R=R_gth, t=t_gth)
    R_init = np.eye(3)
    t_init = np.mean(pcA[:, :3], axis=0)

    # ICP -----
    t, R = icp(pcA, pcB, num_iterations=70, t_init=t_init, R_init=R_init, visualize=visualize)
    print('Infered Position: ', t)
    print('Infered Orientation:', R)
    print('\tReal Position: ', t_gth)
    print('\tReal Orientation:', R_gth)


def real_model_icp(path_to_pointcloud_files, visualize=True):
    # Load the model
    pcB = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med.ply'))  # Model
    pcB[:, 3:] = np.array([.73, .21, .1]) * np.ones((pcB.shape[0], 3)) # Paint it red
    # ------ Noisy partial view scene -----
    pcs, camera_poses = load_pcs_and_camera_poses(path_to_pointcloud_files)
    pc = merge_point_clouds(pcs, camera_poses)
    pcA = filter_point_cloud(pc)
    if visualize:
        print('Displaying filtered point cloud. Close the window to continue.')
        view_point_cloud(pcA)
    else:
        print('Filtered scene point clouds saved as we cannot visualize it.\n Use software such as Meshlab to visualize them.')
        save_point_cloud(pcA, 'filtered_scene_pc', path_to_pointcloud_files)
    R_init = quaternion_matrix(quaternion_from_axis_angle(axis=np.array([0, 0, 1]), angle=np.pi / 2))
    t_init = np.mean(pcA[:, :3], axis=0)
    t_init[-1] = 0
    t, R = custom_icp(pcA, pcB, num_iterations=70, t_init=t_init, R_init=R_init)
    print('Infered Position: ', t)
    print('Infered Orientation:', R)


if __name__ == '__main__':
    # by default we assume that the point cloud files are on the same directory

    path_to_files = 'a4_pointcloud_files' # TODO: Change the path to the directory containing your point cloud files

    # Test for part 1
    # transform_point_cloud_example(path_to_files, visualize=visualize) # TODO: Uncomment to test

    # Test for part 2
    # reconstruct_scene(path_to_files, visualize=visualize) # TODO: Uncomment to test

    # Test for part 5
    # perfect_model_icp(path_to_files, visualize=visualize) # TODO: Uncomment to test

    # Test for part 6
    # real_model_icp(path_to_files, visualize=visualize)    # TODO: Uncomment to test

