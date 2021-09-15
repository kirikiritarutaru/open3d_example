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


if __name__ == '__main__':
    plane_segmentation()
