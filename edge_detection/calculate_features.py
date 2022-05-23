import os
import time
import numpy as np
import pandas as pd
import open3d as o3d
import trimesh

import multiprocessing as mp
from multiprocessing.pool import ThreadPool

from features.helpers import read_roof_cloud, get_roof_folder, normalize_cloud

from features.feature import FeatureState, ScalableFeatureState, SmallScalableFeatureState

from features.lower_voxels import LowerVoxels
from features.upper_voxels import UpperVoxels
from features.edge_voxels import EdgeVoxels
from features.around_voxels import AroundVoxels

from features.knn_centroid_distance import kNNCentroidDistance
from features.normal_cluster import NormalCluster
from features.covariance_eigenvalue import CovarianceEigenvalue

test_file_names = ["32-1-510-215-53-test-1.ply", "32-1-510-215-53-test-2.ply", "32-1-510-215-53-test-3.ply", 
                   "32-1-510-215-53-test-4.ply", "32-1-510-215-53-test-5.ply", "32-1-510-215-53-test-6.ply",
                   "32-1-510-215-53-test-7.ply", "32-1-510-215-53-test-8.ply", "32-1-510-215-53-test-9.ply",
                   "32-1-510-215-53-test-10.ply", "32-1-510-215-53-test-11.ply", "32-1-510-215-53-test-12.ply",
                   "32-1-510-215-53-test-13.ply", "32-1-510-215-53-test-14.ply"]

def get_point_lables(file_name):
  roof_folder = get_roof_folder()
  tri_cloud = trimesh.load(os.path.join(roof_folder, file_name))
  return tri_cloud.metadata["ply_raw"]["vertex"]["data"]["scalar_Classification"]


def store_calculated_values(df: pd.DataFrame, file_name: str):
  file_name_base = file_name.split('.')[0]
  roof_folder = get_roof_folder()
  value_file_name = os.path.join(roof_folder, "calculated_values", file_name_base + ".csv")
  df.to_csv(value_file_name)


def store_labels(all_labels, labels_df, name: str):
  if len(all_labels.shape) == 1:
    # Only global scale of labels
    labels_df[name] = all_labels
  elif len(all_labels.shape) == 2:
    # Several scales of labels
    for label_i in range(all_labels.shape[0]):
      scale_name = name + "_" + str(label_i)
      labels = all_labels[label_i]
      labels_df[scale_name] = labels
    
    # Store mean of all scales as a label as well
    mean_labels = np.divide(np.sum(all_labels, axis=0), all_labels.shape[0])
    labels_df[name + "_mean"] = mean_labels

  elif len(all_labels.shape) == 3:
    # Multifeature:
    for feature_i in range(all_labels.shape[0]):
      feature_i_df = pd.DataFrame()
      # Several scales of labels
      for label_i in range(all_labels.shape[1]):
        scale_name = f"{name}_feature_{feature_i}_scale_{label_i}"
        labels = all_labels[feature_i, label_i]
        feature_i_df[scale_name] = labels
      
      # Store mean of all scales as a label as well
      mean_labels = np.divide(np.sum(all_labels[feature_i], axis=0), all_labels.shape[1])
      mean_name = f"{name}_feature_{feature_i}_mean"
      # labels_df[mean_name] = mean_labels
      mean_df = pd.DataFrame({ mean_name: mean_labels })

      # Concat dataframes
      labels_df = pd.concat([labels_df, feature_i_df, mean_df], axis=1)
  
  return labels_df

def calculate_feature_values(feature, time_store):
      feature_name = feature.__class__.__name__
      print("\tCalculating feature", feature_name)
      tic = time.perf_counter()
      labels = feature.run()
      toc = time.perf_counter()
      time_store[feature_name] = toc - tic
      print(f"\t\tDone calculating feature {feature_name} after {(toc - tic):.4} seconds")
      return labels

def calculate_with_multiprocess(feature_class, feature_state_class, points, time_store):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  feature = feature_class(feature_state_class(pcd))
  return calculate_feature_values(feature, time_store)

def main():
  for file_name in test_file_names:
    print("Calculating features for file:", file_name)
    cloud = read_roof_cloud(file_name)
    cloud = normalize_cloud(cloud)
    points = np.asarray(cloud.points)
    print(f"File contains {points.shape[0]} points.")

    # Add normals to cloud
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=80))
    cloud.orient_normals_consistent_tangent_plane(30)

    # FS = FeatureState(cloud)
    SFS = ScalableFeatureState(cloud)
    SSFS = SmallScalableFeatureState(cloud)

    # Create pandas dataframe to hold all the feature values for each feature and scale
    labels_df = pd.DataFrame()
    
    # Read actual classification label from the cloud, and add it to the dataframe
    labels_df["target"] = get_point_lables(file_name)

    # Add x, y, z values (values between -1 and 1)
    labels_df["x"] = points[:, 0]
    labels_df["y"] = points[:, 1]
    labels_df["z"] = points[:, 2]

    # TODO: Test if adding abs(z) or z^2 would help!
    # The idea is that top and bottom edges can be correlated to z, 
    # and it is easier to separate linearly instead of high and low values
    # Add y^2, as it might help to set one "if" sentence for the z-value. 
    # High and low points will bi high y^2, while middle points will be around 0
    labels_df["z_pow2"] = np.power(points[:, 2], 2)
    labels_df["z_abs"] = np.abs(points[:, 2])

    # Add the mean distance to the points 10 nearest neighbors
    labels_df["10_knn_mean_dist"] = SFS.mean_distances
    labels_df["10_knn_max_dist"] = SFS.max_distances


    # Calculate Fast Point Feature histogram (FPFH)
    '''
    FPFH = o3d.pipelines.registration.compute_fpfh_feature(cloud, o3d.geometry.KDTreeSearchParamKNN()).data
    for fpfh_feature_i in range(FPFH.shape[0]):
      # t = FPFH[fpfh_feature_i, :]
      # print("test shape:", t, t.shape)
      labels_df[f"fpfh_{fpfh_feature_i}"] = FPFH[fpfh_feature_i, :]
    '''

    big_tic = time.perf_counter()
    time_store = {}

    # Use multithreading or multiprocess to speed up the processing time...
    CPU_METHODS = ["linear", "multithread", "multiprocess"]
    CPU_METHOD = CPU_METHODS[2]

    if CPU_METHOD == "multiprocess":
      # Calculate feature classes
      feature_classes = [kNNCentroidDistance, LowerVoxels, UpperVoxels, AroundVoxels, 
                         CovarianceEigenvalue]# , EdgeVoxels]
      feature_state_classes = [ScalableFeatureState, ScalableFeatureState, ScalableFeatureState, ScalableFeatureState, 
                               ScalableFeatureState]#, SmallScalableFeatureState]

      # Initialize multiprocessing pool
      pool = mp.Pool(mp.cpu_count())
      results = pool.starmap(calculate_with_multiprocess, 
                            [(feature_classes[i], feature_state_classes[i], points, time_store) for i in range(len(feature_classes))])

      for result_i, result in enumerate(results):
        feature_name = feature_classes[result_i].__name__
        labels_df = store_labels(result, labels_df, feature_name)
      
      pool.close()

    elif CPU_METHOD == "multithread":
      # Calculate features
      features = [kNNCentroidDistance(SFS), LowerVoxels(SFS), UpperVoxels(SFS), AroundVoxels(SFS), 
                  NormalCluster(SFS), CovarianceEigenvalue(SFS), EdgeVoxels(SSFS)]
      pool = ThreadPool(processes=7)
      results = pool.starmap(calculate_feature_values, [(feature, time_store) for feature in features])

      for result_i, result in enumerate(results):
        feature_name = features[result_i].__name__
        labels_df = store_labels(result, labels_df, feature_name)
      pool.close()

    else:
      # Calculate features
      features = [kNNCentroidDistance(SFS), LowerVoxels(SFS), UpperVoxels(SFS), AroundVoxels(SFS), 
                  NormalCluster(SFS), CovarianceEigenvalue(SFS), EdgeVoxels(SSFS)]
      
      for feature in features:
        result = calculate_feature_values(feature, time_store)
        labels_df = store_labels(result, labels_df, feature.__class__.__name__)
    
    big_toc = time.perf_counter()
    print(f"\tAll features took {big_toc - big_tic:0.4f} seconds to run\n")
    time_store["all_features"] = big_toc - big_tic

    # print(time_store)
    for key, val in time_store.items():
      if key != "all_features":
        part = val/time_store["all_features"]
        print(f"{key} took {val:0.4f} seconds to run. That is {part} out of all")
      else:
        all_f = time_store["all_features"]
        print(f"All features used {all_f} seconds to run")

    print('\n\n\n')
    # print(labels_df.head())
    store_calculated_values(labels_df, file_name)


if __name__ == "__main__":
  main()

'''
Multiprocess:
Calculating features for file: 32-1-510-215-53-test-2.ply
File contains 3734 points.
	Calculating feature kNNCentroidDistance
	Calculating feature LowerVoxels
	Calculating feature AroundVoxels
	Calculating feature UpperVoxels
	Calculating feature NormalCluster
	Calculating feature EdgeVoxels
	Calculating feature CovarianceEigenvalue
		Done calculating feature kNNCentroidDistance after 0.4056 seconds
		Done calculating feature UpperVoxels after 0.4599 seconds
		Done calculating feature LowerVoxels after 0.4716 seconds
		Done calculating feature CovarianceEigenvalue after 1.245 seconds
		Done calculating feature AroundVoxels after 2.558 seconds
		Done calculating feature NormalCluster after 7.191 seconds
		Done calculating feature EdgeVoxels after 15.93 seconds
	All features took 18.2032 seconds to run

Multithread:
Calculating features for file: 32-1-510-215-53-test-2.ply
File contains 3734 points.
	Calculating feature kNNCentroidDistance
	Calculating feature LowerVoxels
	Calculating feature UpperVoxels
	Calculating feature AroundVoxels
	Calculating feature NormalCluster
	Calculating feature CovarianceEigenvalue
	Calculating feature EdgeVoxels
		Done calculating feature kNNCentroidDistance after 0.4396 seconds
		Done calculating feature CovarianceEigenvalue after 6.198 seconds
		Done calculating feature LowerVoxels after 9.622 seconds
		Done calculating feature UpperVoxels after 9.7 seconds
		Done calculating feature NormalCluster after 18.48 seconds
		Done calculating feature AroundVoxels after 20.74 seconds
		Done calculating feature EdgeVoxels after 29.53 seconds
	All features took 29.6477 seconds to run

Linear:
Calculating features for file: 32-1-510-215-53-test-2.ply
File contains 3734 points.
	Calculating feature kNNCentroidDistance
		Done calculating feature kNNCentroidDistance after 0.3816 seconds
	Calculating feature LowerVoxels
		Done calculating feature LowerVoxels after 0.4393 seconds
	Calculating feature UpperVoxels
		Done calculating feature UpperVoxels after 0.4364 seconds
	Calculating feature AroundVoxels
		Done calculating feature AroundVoxels after 2.502 seconds
	Calculating feature NormalCluster
		Done calculating feature NormalCluster after 6.95 seconds
	Calculating feature CovarianceEigenvalue
		Done calculating feature CovarianceEigenvalue after 1.15 seconds
	Calculating feature EdgeVoxels
		Done calculating feature EdgeVoxels after 15.37 seconds
	All features took 27.2367 seconds to run
'''
