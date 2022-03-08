

import os
from turtle import color
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona as fi
import matplotlib.pyplot as plt
import open3d as o3d
import laspy


def read_point_cloud(file_name):
  current_folder = os.path.dirname(os.path.abspath(__file__))
  project_folder = os.path.dirname(current_folder)
  data_folder = os.path.join(project_folder, "data/point_clouds/raw")
  cloud_path = os.path.join(data_folder, file_name)

  print("Reading data...")

  cloud = laspy.read(cloud_path)

  print("Removing vegetation and bridges...")

  cloud = cloud[cloud.classification == 1] # Only unclassified points

  points = np.vstack((cloud.x, cloud.y, cloud.z)).transpose()
  colors = np.vstack((cloud.red, cloud.green, cloud.blue)).transpose()

  maximum_color = np.max([np.max(colors[:, 0]), np.max(colors[:, 0]), np.max(colors[:, 0])])

  colors = colors/maximum_color

  pcd = o3d.geometry.PointCloud()
  print("Setting points...")
  pcd.points = o3d.utility.Vector3dVector(points)

  print("Setting colors...")
  pcd.colors = o3d.utility.Vector3dVector(colors)

  # o3d.visualization.draw_geometries([pcd])

  print('Preprocessing done!')

  return pcd

def read_building_data(file_name):
  current_folder = os.path.dirname(os.path.abspath(__file__))
  project_folder = os.path.dirname(current_folder)
  building_path = os.path.join(project_folder, "data/building_footprints", file_name)
  return gpd.read_file(building_path, layer = "fkb_bygning_omrade")

def write_cropped_cloud_result(file_name, cloud):
  current_folder = os.path.dirname(os.path.abspath(__file__))
  project_folder = os.path.dirname(current_folder)
  cropped_folder = os.path.join(project_folder, "data/point_clouds/cropped")

  # Create folder if not exists
  if len(file_name.split("/")):
    cloud_folder = os.path.join(cropped_folder, file_name.split("/")[0])
    if not os.path.exists(cloud_folder):
      os.makedirs(cloud_folder)


  return o3d.io.write_point_cloud(os.path.join(cropped_folder, file_name), cloud, print_progress=True)

def write_roof_cloud_result(file_name, cloud):
  current_folder = os.path.dirname(os.path.abspath(__file__))
  project_folder = os.path.dirname(current_folder)
  cropped_folder = os.path.join(project_folder, "data/point_clouds/roofs")

  # Create folder if not exists
  if len(file_name.split("/")):
    cloud_folder = os.path.join(cropped_folder, file_name.split("/")[0])
    if not os.path.exists(cloud_folder):
      os.makedirs(cloud_folder)


  return o3d.io.write_point_cloud(os.path.join(cropped_folder, file_name), cloud, print_progress=True)

def crop_building_data(buildings, BB):
  corner_points = np.asarray(BB.get_box_points())

  max_x = corner_points[np.argmax(corner_points[:,0])][0]
  min_x = corner_points[np.argmin(corner_points[:,0])][0]
  max_y = corner_points[np.argmax(corner_points[:,1])][1]
  min_y = corner_points[np.argmin(corner_points[:,1])][1]

  return buildings.cx[min_x:max_x, min_y:max_y]

def crop_point_cloud(cloud, bounds):
  # Building data x and y bounds
  minx, miny, maxx, maxy = bounds

  # Point cloud z bounds
  cloud_BB = cloud.get_axis_aligned_bounding_box()
  corner_points = np.asarray(cloud_BB.get_box_points())
  maxz = corner_points[np.argmax(corner_points[:,2])][2]
  minz = corner_points[np.argmin(corner_points[:,2])][2]

  bounding_polygon = np.zeros((8,3))
  bounding_polygon[0] = [minx, miny, minz]
  bounding_polygon[1] = [minx, maxy, minz]
  bounding_polygon[2] = [maxx, miny, minz]
  bounding_polygon[3] = [maxx, maxy, minz]

  bounding_polygon[4] = [minx, miny, maxz]
  bounding_polygon[5] = [minx, maxy, maxz]
  bounding_polygon[6] = [maxx, miny, maxz]
  bounding_polygon[7] = [maxx, maxy, maxz]

  vol = o3d.visualization.SelectionPolygonVolume()
  vol.orthogonal_axis = "Y"
  vol.axis_max = maxy
  vol.axis_min = miny
  bounding_polygon[:, 1] = 0
  vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

  return vol.crop_point_cloud(cloud)


# Once the building is cut out, run a DBScan on it, remove the group with lowest z-value
# and then only keep the group with the most points (aka ground is removed and cannot be the largest)

# See if it is necessary to do some outlier removal after that or not. 
# It is maybe best to do it before the DBScan...
def extract_roof_from_building(cloud):
  labels = np.array(cloud.cluster_dbscan(eps=0.5, min_points=100))

  points = np.asarray(cloud.points)
  colors = np.asarray(cloud.colors)

  clusters = np.unique(labels)
  mean_z = np.zeros(clusters.shape[0] - 1) # Do not count -1 as a cluster as it is noise
  for cluster in clusters:
    if cluster != -1:
      points_in_cluster = points[labels == cluster]
      mean_z[cluster] = np.mean(points_in_cluster[:,2])

  roof_cluster = np.argmax(mean_z)

  roof = o3d.geometry.PointCloud()
  roof.points = o3d.utility.Vector3dVector(points[labels == roof_cluster])
  roof.colors = o3d.utility.Vector3dVector(colors[labels == roof_cluster])

  return roof



# TODO: Read different point clouds
# Read point cloud and FKB-Bygning data
cloud_file_name_base = "32-1-510-215-53"
cloud_file_name = cloud_file_name_base + ".laz"
building_file_name = "Basisdata_5001_Trondheim_5972_FKB-Bygning_FGDB.gdb"

cloud = read_point_cloud(cloud_file_name)
buildings = read_building_data(building_file_name)

# Clip the FKB_Bygning data to the BB of the point cloud
# Use the .cx[xmin:xmax, ymin:ymax] indexing to do so
BB = cloud.get_axis_aligned_bounding_box()
buildings = crop_building_data(buildings, BB)


# Loop through the ramaining buildings in FKB-Bygning data
# Use the building geometry, add a buffer, and make a BB for the point cloud
# Use the lowest and the highest points in the point cloud as the z-limits in the BB

print("Finding and saving roofs...")

count = 0
for i in buildings.index:
  # print(buildings[buildings.index == buildings.index[i]])
  building = buildings[buildings.index == i]
  geometry = building["geometry"]
  buffered = geometry.buffer(2)

  building_cloud = crop_point_cloud(cloud, buffered.total_bounds)

  # o3d.visualization.draw_geometries([building_cloud])

  # roof_cloud = extract_roof_from_building(building_cloud)

  # o3d.visualization.draw_geometries([roof_cloud])

  points = np.asarray(building_cloud.points)
  
  # Save the cropped point cloud in data/point_cloud/roofs/{original point cloud name}-{roof index}.ply
  new_cloud_file_name = cloud_file_name_base + "/" + str(i) + "-" + str(points.shape[0]) + ".ply"
  print("Saving roof", i, "as:", new_cloud_file_name)

  write_cropped_cloud_result(new_cloud_file_name, building_cloud)
  # write_roof_cloud_result(new_cloud_file_name, roof_cloud)
  
  count += 1

  if count >= 20:
    break


print("Done!")




