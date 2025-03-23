import numpy as np
import pandas as pd
import os
import math
import copy
try:
    import open3d as o3d
except ImportError:
    pass


# ===========================================================================

# AUXILIARY FUNCTIONS (DO NOT CHANGE)


class ICPVisualizer(object):

    def __init__(self, pcA, pcB):
        self.pcA = pcA # scene
        self.pcB = pcB # model
        self.pcB_tr = copy.deepcopy(self.pcB)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.scene = get_o3d_pc(self.pcA)
        self.model = get_o3d_pc(self.pcB)
        self.line_set = o3d.geometry.LineSet()

        self.vis.add_geometry(self.scene)
        self.vis.add_geometry(self.model)
        self.vis.add_geometry(self.line_set)
        self.vis.poll_events()
        self.vis.update_renderer()

    def _set_zero_line_set(self):
        empty_line_set = o3d.geometry.LineSet()
        self.line_set.points = o3d.utility.Vector3dVector()
        self.line_set.colors = o3d.utility.Vector3dVector()
        # line_set.lines = o3d.utility.Vector3dVector()
        self.line_set.lines = empty_line_set.lines
        self.vis.update_geometry(self.line_set)

    def view_icp(self, R, t):
        self._set_zero_line_set()
        self.pcB_tr = self.__tr_pc(self.pcB, R=R, t=t)
        self.model.points = o3d.utility.Vector3dVector(self.pcB_tr[:, :3])
        self.vis.update_geometry(self.model)
        self.vis.poll_events()
        self.vis.update_renderer()

    def plot_correspondences(self, correspondences):
        model_points_selected_tr = self.pcB_tr[correspondences]
        new_line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(self.scene,
                                                                                get_o3d_pc(model_points_selected_tr),
                                                                                [(i, i) for i in
                                                                                 range(len(correspondences))])
        if self.line_set is None:
            self.line_set = new_line_set
            self.vis.add_geometry(self.line_set)
        else:
            self.line_set.points = new_line_set.points
            self.line_set.lines = new_line_set.lines
            self.line_set.colors = new_line_set.colors
            self.vis.update_geometry(self.line_set)
        self.vis.poll_events()
        self.vis.update_renderer()

    def __tr_pc(self, pc, R, t):
        XYZs = pc[:, :3]
        XYZ_tr = np.matmul(R, XYZs.T).T + t
        tr_pc = np.concatenate([XYZ_tr, pc[:, 3:]], axis=-1)
        return tr_pc


def save_point_cloud(pc, name, save_path):
    """
    :param pc: Point cloud as an array (N,6), where last dim is as:
        - X Y Z R G B
    :param name:
    :return:
    """
    num_points = pc.shape[0]
    point_lines = []
    pc_color = pc[:, 3:]
    if np.all(pc_color <= 1):
        pc[:, 3:] *= 255
    for point in pc:
        point_lines.append(
            "{:f} {:f} {:f} {:d} {:d} {:d} 255\n".format(point[0], point[1], point[2], int(point[3]), int(point[4]),
                                                         int(point[5])))
    points_text = "".join(point_lines)
    file_name = '{}.ply'.format(name)
    pc_path = os.path.join(save_path, file_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open(pc_path, 'w+') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(num_points))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('property uchar alpha\n')
        f.write('end_header\n')
        f.write(points_text)

    print('PC saved as {}'.format(pc_path))


def view_point_cloud(pc):
    try:
        pcds = []
        if type(pc) is not list:
            pc = [pc]
        for pc_i in pc:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_i[:,:3])
            pcd.colors = o3d.utility.Vector3dVector(pc_i[:,3:7])
            pcds.append(pcd)
        o3d.visualization.draw_geometries(pcds)
    except NameError:
        print('No o3d was found -- \n\tInstall Open3d or visualize the saved point cloud (as .ply) using MeshLab')


def view_point_cloud_from_file(file_path):
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        o3d.visualization.draw_geometries([pcd])
    except NameError:
        print('No o3d was found -- \n\tInstall Open3d or visualize the saved point cloud (as .ply) using MeshLab')


def load_point_cloud(ply_file_path):
    pc = None
    try:
        pcd = o3d.io.read_point_cloud(ply_file_path)
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)
        pc = np.concatenate([pcd_points, pcd_colors], axis=1)
        print('PC SHAPE: ', pc.shape) # TODO: Remove
    except NameError:
        print('No o3d was found -- \n\tInstall Open3d or visualize the saved point cloud (as .ply) using MeshLab')
    return pc


def get_o3d_pc(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:7])
    return pcd


def quaternion_matrix(quaternion):
    """Return rotation matrix from quaternion.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    _EPS = np.finfo(float).eps * 4.0
    if n < _EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])


def quaternion_from_axis_angle(axis, angle):
    qw = np.cos(angle/2)*np.ones(1)
    quat = np.concatenate([qw, np.sin(angle/2)*axis])
    return quat


def load_pcs_and_camera_poses(path_to_files):
    cp_path = os.path.join(path_to_files, 'camera_poses.csv')
    # Using pandas to load the csv file: --
    # camera_poses_df = pd.read_csv(cp_path)
    # camera_poses = [(line[2:5].astype(np.float), quaternion_matrix(line[5:].astype(np.float))) for line in
    #                 camera_poses_df.values]

    # Using numpy to load the csv file
    _camera_poses_array = np.genfromtxt(cp_path, delimiter=',')
    camera_poses_array = _camera_poses_array[1:,2:] # remove header and other not useful values
    camera_poses = [(line[:3].astype(np.float64), quaternion_matrix(line[3:].astype(np.float64))) for line in camera_poses_array]
    num_cameras = len(camera_poses)

    # Load pointcloud from each camera
    pcs = [load_point_cloud(os.path.join(path_to_files, 'test_multiple_objects_pc_camera_{}.ply'.format(i))) for i in
           range(num_cameras)]

    return pcs, camera_poses


def load_object_poses(path_to_files):
    op_path = os.path.join(path_to_files, 'simulation_data.csv')
    df = pd.read_csv(op_path)
    object_names = df['object_name']
    object_poses = {}
    for i, object_name in enumerate(object_names):
        pos_i = _process_array(df['object_position'][i])
        quat_i = _process_array(df['object_quat'][i])
        object_poses[object_name] = {'pos': pos_i, 'quat': quat_i}
    return object_poses


def _process_array(str_array):
    _str_array = str_array[1:-1]# remove the [ and ] at the ends
    str_array_list = _str_array.split(' ')
    array_list = [float(i) for i in str_array_list if i not in ['', ' ']]
    processed_array = np.array(array_list)
    return processed_array