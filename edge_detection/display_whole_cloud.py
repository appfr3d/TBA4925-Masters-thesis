from turtle import width
import open3d as o3d
pcd = o3d.io.read_point_cloud("../data/point_clouds/raw/32-1-510-215-53.ply")
o3d.visualization.draw_geometries([pcd], width=1000, height=1000)