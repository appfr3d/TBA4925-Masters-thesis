import os

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


def get_cloud_file_name():
	for i in range(1, 6):
		print("(" + str(i) + ")")
	chosen_index = int(input("> "))

	while chosen_index < 1 or chosen_index >= len(feature_names):
		print("Must be between 1 og " + str(len(feature_names) - 1))
		chosen_index = int(input("> "))

	return "32-1-510-215-53-test-" + str(chosen_index) + ".ply"

# TODO: enable testing of all features and all test-files

print("Which feature do you want to test?:")
feature_index = get_feature_index()

print("Which cloud file do you want to test?:")
file_name = get_cloud_file_name()
cloud = read_roof_cloud(file_name)
cloud = normalize_cloud(cloud)
print(cloud)

state = all_features[feature_index][1](cloud)
f = all_features[feature_index][0](state)
f.run_test(all_features[feature_index][2], file_name)