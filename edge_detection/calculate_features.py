import os
import numpy as np
import pandas as pd
import open3d as o3d
import trimesh

from features.helpers import read_roof_cloud, get_roof_folder, normalize_cloud

from features.feature import FeatureState, ScalableFeatureState, SmallScalableFeatureState

from features.lower_voxels import LowerVoxels
from features.upper_voxels import UpperVoxels
from features.edge_voxels import EdgeVoxels
from features.around_voxels import AroundVoxels

from features.knn_centroid_distance import kNNCentroidDistance
from features.normal_cluster import NormalCluster
from features.covariance_eigenvalue import CovarianceEigenvalue

test_file_names = ["32-1-510-215-53-test-4.ply", "32-1-510-215-53-test-5.ply"] 

def get_point_lables(file_name):
  roof_folder = get_roof_folder()
  tri_cloud = trimesh.load(os.path.join(roof_folder, file_name))
  return tri_cloud.metadata["ply_raw"]["vertex"]["data"]["scalar_Classification"]


def store_calculated_values(df: pd.DataFrame, file_name: str):
  file_name_base = file_name.split('.')[0]
  roof_folder = get_roof_folder()
  value_file_name = os.path.join(roof_folder, "calculated_values", file_name_base + ".csv")
  df.to_csv(value_file_name)


def main():
  for file_name in test_file_names:
    cloud = read_roof_cloud(file_name)
    cloud = normalize_cloud(cloud)
    print(cloud)
    points = np.asarray(cloud.points)

    FS = FeatureState(cloud)
    SFS = ScalableFeatureState(cloud)
    SSFS = SmallScalableFeatureState(cloud)

    # Create pandas dataframe to hold all the feature values for each feature and scale
    labels_df = pd.DataFrame()
    
    def store_labels(all_labels, name: str):
      if len(all_labels.shape) == 1:
        # Only global scale of labels
        labels_df[name] = all_labels
      else:
        # Several scales of labels
        for label_i in range(all_labels.shape[0]):
          scale_name = name + '_' + str(label_i)
          labels = all_labels[label_i]
          labels_df[scale_name] = labels

    # Read actual classification label from the cloud, and add it to the dataframe
    labels_df["target"] = get_point_lables(file_name)

    # Add x, y, z values (values between -1 and 1)
    labels_df["x"] = points[:, 0]
    labels_df["y"] = points[:, 1]
    labels_df["z"] = points[:, 2]

    # TODO: Test if adding abs(z) or z^2 would help! 
    # The idea is that top and bottom edges can be correlated to z, 
    # and it is easier to separate linearly instead of high and low values

    # Calculate features
    features = [kNNCentroidDistance(FS), LowerVoxels(SFS), UpperVoxels(SFS), AroundVoxels(SFS), 
                NormalCluster(SFS), CovarianceEigenvalue(SFS), EdgeVoxels(SSFS)]

    for feature in features:
      feature_name = feature.__class__.__name__
      print("Calculating feature", feature_name)
      labels = feature.run()
      store_labels(labels, feature_name)

    # print(labels_df.head())
    store_calculated_values(labels_df, file_name)


if __name__ == "__main__":
  main()
