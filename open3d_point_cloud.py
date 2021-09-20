import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def viz_pcd(pcd: o3d.cpu.pybind.geometry.PointCloud):
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries(
        [pcd],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024]
    )
    # -- Mouse view control --
    #   Left button + drag         : Rotate.
    #   Ctrl + left button + drag  : Translate.
    #   Wheel button + drag        : Translate.
    #   Shift + left button + drag : Roll.
    #   Wheel                      : Zoom in/out.

    # -- Keyboard view control --
    #   [/]          : Increase/decrease field of view.
    #   R            : Reset view point.
    #   Ctrl/Cmd + C : Copy current view status into the clipboard.
    #   Ctrl/Cmd + V : Paste view status from clipboard.

    # -- General control --
    #   Q, Esc       : Exit window.
    #   H            : Print help message.
    #   P, PrtScn    : Take a screen capture.
    #   D            : Take a depth capture.
    #   O            : Take a capture of current rendering settings.


def voxel_downsampling(pcd: o3d.cpu.pybind.geometry.PointCloud):
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries(
        [downpcd],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024]
    )


def vertex_normal_estimation(pcd: o3d.cpu.pybind.geometry.PointCloud):
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        )
    )
    o3d.visualization.draw_geometries(
        [downpcd],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024],
        point_show_normal=True
    )


def crop_point_cloud(
    pcd: o3d.cpu.pybind.geometry.PointCloud,
    paint_color: bool = False
):
    vol = o3d.visualization.read_selection_polygon_volume('cropped.json')
    chair = vol.crop_point_cloud(pcd)
    if paint_color:
        chair.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries(
        [chair],
        zoom=0.7,
        front=[0.5439, -0.2333, -0.8060],
        lookat=[2.4615, 2.1331, 1.338],
        up=[-0.1781, -0.9708, 0.1608]
    )


# Ctrl+C で視点を文字列（json形式）にしてクリップボードに保存←便利すぎ!!!!
# Ctrl+V でクリップボードに保存された視点を再現
def play_with_pcd():
    pcd = o3d.io.read_point_cloud('fragment.ply')
    viz_pcd(pcd)
    voxel_downsampling(pcd)
    vertex_normal_estimation(pcd)
    crop_point_cloud(pcd, True)


def plane_segmentation():
    pcd = o3d.io.read_point_cloud('fragment.pcd')
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    [a, b, c, d] = plane_model
    print(f'Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0')
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries(
        [inlier_cloud, outlier_cloud],
        zoom=0.8,
        front=[-0.4999, -0.1659, -0.8499],
        lookat=[2.1813, 2.0619, 2.0999],
        up=[0.1204, -0.9852, 0.1215]
    )


def point_cluod_distance():
    pcd = o3d.io.read_point_cloud('fragment.ply')
    vol = o3d.visualization.read_selection_polygon_volume('cropped.json')
    chair = vol.crop_point_cloud(pcd)

    # fragment.plyの中の点群と椅子の点群の距離を計算
    dists = pcd.compute_point_cloud_distance(chair)  # ２つの点群の面取り距離も取れるね
    # 上記の距離が0.01より大きい点のインデックスを抽出
    dists = np.asarray(dists)
    ind = np.where(dists > 0.01)[0]
    pcd_without_chair = pcd.select_by_index(ind)

    o3d.visualization.draw_geometries(
        [pcd_without_chair],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024]
    )


def DBSCAN_clustering():
    pcd = o3d.io.read_point_cloud('fragment.ply')
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(
            pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)
        )
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters!")
    colors = plt.get_cmap('tab20')(labels/(max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries(
        [pcd],
        zoom=0.455,
        front=[-0.4999, -0.1659, -0.8499],
        lookat=[2.1813, 2.0619, 2.0999],
        up=[0.1204, -0.9852, 0.1215]
    )


def read_camera_intrinsic():
    pinhole_camera_intrinsic = (
        o3d.io.read_pinhole_camera_intrinsic('camera_primesense.json')
    )
    print(pinhole_camera_intrinsic.intrinsic_matrix)


def custom_draw_geometry_with_custom_fov(pcd, fov_step):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    ctr.change_field_of_view(step=fov_step)
    print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()


def play_custom_fov():
    pcd = o3d.io.read_point_cloud('fragment.ply')
    custom_draw_geometry_with_custom_fov(pcd, 0.0)


def geometry_primitives():
    print("Let's define some primitives")
    mesh_box = o3d.geometry.TriangleMesh.create_box(
        width=1.0, height=1.0, depth=1.0
    )
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.3,
        height=4.0)
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[-2, -2, -2]
    )

    print("We draw a few primitives using collection.")
    o3d.visualization.draw_geometries(
        [mesh_box, mesh_sphere, mesh_cylinder, mesh_frame]
    )

    print("We draw a few primitives using + operator of mesh.")
    o3d.visualization.draw_geometries(
        [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame]
    )


if __name__ == '__main__':
    # play_with_pcd()
    # play_custom_fov()
    geometry_primitives()
