import os
import numpy as np
from features.feature import FeatureState, ScalableFeatureState, SmallScalableFeatureState


from features.lower_voxels import LowerVoxels
from features.upper_voxels import UpperVoxels
from features.edge_voxels import EdgeVoxels
from features.around_voxels import AroundVoxels

from features.knn_centroid_distance import kNNCentroidDistance
from features.normal_cluster import NormalCluster
from features.covariance_eigenvalue import CovarianceEigenvalue

from features.helpers import read_roof_cloud, normalize_cloud

all_features = [(kNNCentroidDistance, FeatureState, "knn_centroid_distance"), 
                (LowerVoxels, ScalableFeatureState, "lower_voxels"), 
                (UpperVoxels, ScalableFeatureState, "upper_voxels"), 
                (AroundVoxels, ScalableFeatureState, "around_voxels"), 
                (NormalCluster, ScalableFeatureState, "normal_cluster"), 
                (CovarianceEigenvalue, ScalableFeatureState, "covariance_eigenvalue"), 
                (EdgeVoxels, SmallScalableFeatureState, "edge_voxels")]

feature_names = list(map(lambda f: f[0].__name__, all_features))

def get_feature_index():
  for file_i, file in enumerate(feature_names):
    print("(" + str(file_i) + ") " + file)
  chosen_index = int(input("> "))

  while chosen_index < 0 or chosen_index >= len(feature_names):
    print("Must be between 0 og " + str(len(feature_names) - 1))
    chosen_index = int(input("> "))

  return chosen_index


def get_cloud_index():
  for i in range(1, 6):
    print("(" + str(i) + ")")
  print("(-1) All clouds")
  chosen_index = int(input("> "))

  while chosen_index != -1 and (chosen_index < 1 or chosen_index >= len(feature_names)):
    print("Must be between 1 og " + str(len(feature_names) - 1))
    chosen_index = int(input("> "))

  return chosen_index

def get_z_rotation_angle():
  angle_names = ["0", "PI/2", "PI", "3*PI/2"]
  angles = [0, np.pi/2, np.pi, 3*np.pi/2]
  for angle_i, angle in enumerate(angle_names):
    print("(" + str(angle_i) + ") " + str(angle))
  chosen_index = int(input("> "))

  while chosen_index < 0 or chosen_index >= len(angle_names):
    print("Must be between 0 og " + str(len(angle_names) - 1))
    chosen_index = int(input("> "))

  return angles[chosen_index]

def test_feature_on_cloud(feature_index, cloud_file_name, angle):
  print("Testing out", all_features[feature_index][2], "on", cloud_file_name)
  cloud = read_roof_cloud(cloud_file_name)
  cloud = normalize_cloud(cloud)
  if angle != 0:
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    rotation =  np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0,0,1]])
    cloud.rotate(rotation)
    parts = cloud_file_name.split('.')
    cloud_file_name = parts[0] + "-" + str(round(np.rad2deg(angle))) + "." + parts[1]
  print(cloud)

  state = all_features[feature_index][1](cloud)
  f = all_features[feature_index][0](state)
  f.run_test(all_features[feature_index][2], cloud_file_name)


# TODO: enable testing of all features and all test-files
print("Which feature do you want to test?:")
feature_index = get_feature_index()

print("Which cloud file do you want to test?:")
cloud_index = get_cloud_index()

print("At which rotation do you want to test?:")
angle = get_z_rotation_angle()

if cloud_index == -1:
  for i in range(1, 6):
    cloud_file_name = "32-1-510-215-53-test-" + str(i) + ".ply"
    test_feature_on_cloud(feature_index, cloud_file_name, angle)

else:
  cloud_file_name = "32-1-510-215-53-test-" + str(cloud_index) + ".ply"
  test_feature_on_cloud(feature_index, cloud_file_name, angle)
