import os
import numpy as np
import pandas as pd
import open3d as o3d
from torch import threshold
import trimesh
import time
import multiprocessing as mp
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm

from catboost import CatBoostClassifier, Pool

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from features.feature import ScalableFeatureState, SmallScalableFeatureState

from features.lower_voxels import LowerVoxels
from features.upper_voxels import UpperVoxels
from features.edge_voxels import EdgeVoxels
from features.around_voxels import AroundVoxels

from features.knn_centroid_distance import kNNCentroidDistance
from features.normal_cluster import NormalCluster
from features.covariance_eigenvalue import CovarianceEigenvalue

from features.helpers import get_roof_folder, read_roof_cloud, normalize_cloud, get_project_folder, remove_noise

all_features = [(kNNCentroidDistance, ScalableFeatureState, "knn_centroid_distance"), 
                (LowerVoxels, ScalableFeatureState, "lower_voxels"), 
                (UpperVoxels, ScalableFeatureState, "upper_voxels"), 
                (AroundVoxels, ScalableFeatureState, "around_voxels"), 
                (NormalCluster, ScalableFeatureState, "normal_cluster"), 
                (CovarianceEigenvalue, ScalableFeatureState, "covariance_eigenvalue"), 
                (EdgeVoxels, SmallScalableFeatureState, "edge_voxels")]

feature_names = list(map(lambda f: f[0].__name__, all_features))


test_file_names = ["32-1-510-215-53-test-1.ply", "32-1-510-215-53-test-2.ply", "32-1-510-215-53-test-3.ply", 
                   "32-1-510-215-53-test-4.ply", "32-1-510-215-53-test-5.ply", "32-1-510-215-53-test-6.ply",
                   "32-1-510-215-53-test-7.ply", "32-1-510-215-53-test-8.ply", "32-1-510-215-53-test-9.ply",
                   "32-1-510-215-53-test-10.ply", "32-1-510-215-53-test-11.ply", "32-1-510-215-53-test-12.ply",
                   "32-1-510-215-53-test-13.ply", "32-1-510-215-53-test-14.ply"]

training_data_file_names = ["32-1-510-215-53-test-1.csv", "32-1-510-215-53-test-2.csv", "32-1-510-215-53-test-3.csv", 
                            "32-1-510-215-53-test-4.csv", "32-1-510-215-53-test-5.csv", "32-1-510-215-53-test-6.csv",
                            "32-1-510-215-53-test-9.csv", "32-1-510-215-53-test-12.csv", "32-1-510-215-53-test-14.csv"]

evaluation_data_file_names = ["32-1-510-215-53-test-10.csv", "32-1-510-215-53-test-11.csv", "32-1-510-215-53-test-8.csv",
                              "32-1-510-215-53-test-13.csv", "32-1-510-215-53-test-7.csv"]


def get_main_menu_choice():
  print("\n\nMain Menu:")
  choices = ["Calculate features", "Classify points", "Calculate and classify", "Test feature removal", "Test feature", "Plot feature figure", "Calculate feature performance"]
  for choice_i, choice in enumerate(choices):
    print("(" + str(choice_i) + ") " + str(choice))
  print("(-1) Quit")
  chosen_index = int(input("> "))

  while chosen_index != -1 and (chosen_index < 0 or chosen_index >= len(choices)):
    print("Must be between 0 and " + str(len(choices) - 1))
    chosen_index = int(input("> "))

  return chosen_index

def get_dataset_path(file_name):
  project_folder = get_project_folder()
  # Could be os.path.join(project_folder, "edge_detection/calculated_values", file_name)
  return os.path.join(project_folder, "data/point_clouds/classified_roofs/calculated_values", file_name)

def calculate_weight(targets):
  one_targets = sum([1 if x == 1 else 0 for x in targets])
  zero_targets = sum([1 if x == 0 else 0 for x in targets])
  one_weight = zero_targets/(one_targets+zero_targets)
  zero_weight = one_targets/(one_targets+zero_targets)
  return [one_weight if target == 1 else zero_weight for target in targets]

def get_point_lables(file_name):
  tri_cloud = trimesh.load(os.path.join(get_roof_folder(), file_name))
  return tri_cloud.metadata["ply_raw"]["vertex"]["data"]["scalar_Classification"]

def calculate_feature_values(feature):
    feature_name = feature.__class__.__name__
    print("\tCalculating feature", feature_name)
    tic = time.perf_counter()
    labels = feature.run()
    toc = time.perf_counter()
    print(f"\t\tDone calculating feature {feature_name} after {(toc - tic):.4} seconds")
    return labels

def calculate_with_multiprocess(feature_class, feature_state_class, points):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  feature = feature_class(feature_state_class(pcd))
  return calculate_feature_values(feature)

def calculate_features():
  def store_labels(all_labels, labels_df, name: str):
    if len(all_labels.shape) == 1:
      # Only global scale of labels
      labels_df[name] = all_labels
    elif len(all_labels.shape) == 2:
      # Several scales of labels
      for scale_i in range(all_labels.shape[0]):
        scale_name = name + "_" + str(scale_i)
        labels = all_labels[scale_i]
        labels_df[scale_name] = labels
      
      # Store mean of all scales as a label as well
      mean_labels = np.mean(all_labels, axis=0)
      # np.divide(np.sum(all_labels, axis=0), all_labels.shape[0])
      labels_df[name + "_mean"] = mean_labels

    elif len(all_labels.shape) == 3:
      # Multifeature:
      # [scale, feature, point_index]
      for feature_i in range(all_labels.shape[1]):
        feature_i_df = pd.DataFrame()
        # Several scales of labels
        for scale_i in range(all_labels.shape[0]):
          scale_name = f"{name}_feature_{feature_i}_scale_{scale_i}"
          labels = all_labels[scale_i, feature_i]
          feature_i_df[scale_name] = labels
        
        # Store mean of all scales as a label as well
        mean_labels = np.mean(all_labels[:, feature_i], axis=0)
        # np.divide(np.sum(all_labels[scale_i], axis=0), all_labels.shape[1])
        mean_name = f"{name}_feature_{feature_i}_mean"
        # labels_df[mean_name] = mean_labels
        mean_df = pd.DataFrame({ mean_name: mean_labels })

        # Concat dataframes
        labels_df = pd.concat([labels_df, feature_i_df, mean_df], axis=1)
    
    return labels_df
  
  for file_name in test_file_names:
    print("Calculating features for file:", file_name)
    cloud = read_roof_cloud(file_name)
    cloud, downsampling_factor = normalize_cloud(cloud)
    cloud, indices = remove_noise(cloud)
    points = np.asarray(cloud.points)
    print(f"File contains {points.shape[0]} points.")

    # Create pandas dataframe to hold all the feature values for each feature and scale
    labels_df = pd.DataFrame()
    
    # Read actual classification label from the cloud, and add it to the dataframe
    labels_df["target"] = get_point_lables(file_name)[indices]

    # Add x, y, z, z_pow2 and z_abs
    labels_df["x"] = points[:, 0]
    labels_df["y"] = points[:, 1]
    labels_df["z"] = points[:, 2]
    labels_df["z_pow2"] = np.power(points[:, 2], 2)
    labels_df["z_abs"] = np.abs(points[:, 2])

    # Add the mean distance to the points 10 nearest neighbors
    SFS = ScalableFeatureState(cloud, downsampling_factor)
    labels_df["10_knn_mean_dist"] = SFS.mean_distances * downsampling_factor
    labels_df["10_knn_max_dist"] = SFS.max_distances * downsampling_factor

    big_tic = time.perf_counter()

    # Calculate feature classes
    feature_classes = [kNNCentroidDistance, LowerVoxels, UpperVoxels, AroundVoxels, CovarianceEigenvalue, EdgeVoxels]
    feature_state_classes = [ScalableFeatureState, ScalableFeatureState, ScalableFeatureState, ScalableFeatureState, ScalableFeatureState, SmallScalableFeatureState]

    # Initialize multiprocessing pool
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(calculate_with_multiprocess, 
                          [(feature_classes[i], feature_state_classes[i], points) for i in range(len(feature_classes))])
    
    for result_i, result in enumerate(results):
      feature_name = feature_classes[result_i].__name__
      # TODO: just concatenate features here
      labels_df = store_labels(result, labels_df, feature_name)
    
    pool.close()
    
    big_toc = time.perf_counter()
    print(f"\tAll features took {big_toc - big_tic:0.4f} seconds to run\n\n\n\n")

    # Store calculated values
    file_name_base = file_name.split('.')[0]
    roof_folder = get_roof_folder()
    value_file_name = os.path.join(roof_folder, "calculated_values", file_name_base + ".csv")
    labels_df.to_csv(value_file_name)

def classify_points():
  # TODO: Ask to load model or train new
  # TODO: Ask for grid search

  training_data = pd.DataFrame()
  for file_name in training_data_file_names:
    data = pd.read_csv(get_dataset_path(file_name), index_col=0)
    training_data = pd.concat([training_data, data])

  # REMOVE FEATURES
  # Remove target from features
  all_features = training_data.drop("target", axis=1)
  feature_list = all_features.columns.tolist()

  # Remove features
  remove_group_features = ['EdgeVoxels', 'x', 'y', '10', 'CovarianceEigenvalue_feature_6'] #"LowerVoxels"]'CovarianceEigenvalue_feature_6'
  remove_single_features = ['z', 'z_abs'] #  '10_knn_mean_dist'

  remove_features = []
  for feature_group_name in remove_group_features:  
    remove_features += [f for f in feature_list if f.startswith(feature_group_name)]  # Remove groups
  remove_features += [f for f in remove_single_features if f not in remove_features]  # Remove singles

  print("Classifying without features:", remove_features)

  # TRAIN
  # Split in traning and testing splits
  features = training_data.drop(["target"] + remove_features, axis=1)
  targets = training_data["target"]
  X_train, X_validation, y_train, y_validation = train_test_split(features, targets, train_size=0.8, random_state=1234)

  # Weight the edge / non-edge classes
  target_weight = calculate_weight(y_train)
  train_pool = Pool(data=X_train, label=y_train, weight=target_weight)
  model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    random_seed=42,
    use_best_model=True,
    logging_level="Silent",
  )
  model.fit(train_pool, eval_set=(X_validation, y_validation), verbose=True)

  # Calculate feature importance
  feature_importance_list = model.get_feature_importance(type="FeatureImportance")
  feature_list = features.columns.tolist()
  feature_importance = pd.DataFrame({"feature": feature_list, "importance": feature_importance_list})
  feature_importance.sort_values(by=["importance"], ascending=[False], inplace=True)
  print("\nFeature importance:")
  pd.set_option('display.max_rows', len(feature_list))
  print(feature_importance)

  # EVALUATE
  evaluation_data = pd.DataFrame()
  for file_name in evaluation_data_file_names:
    data = pd.read_csv(get_dataset_path(file_name), index_col=0)
    evaluation_data = pd.concat([evaluation_data, data])

  evaluation_features = evaluation_data.drop(["target"] + remove_features, axis=1)
  targets = evaluation_data["target"].to_numpy()
  predictions = model.predict(evaluation_features)

  # Evaluation metrics
  # iou and f1 formulas from: https://tomkwok.com/posts/iou-vs-f1/
  print("Evaluation:")
  point_percent = np.sum(np.equal(predictions, targets))/predictions.shape[0]
  precision = precision_score(targets, predictions)
  recall = recall_score(targets, predictions)
  iou = (precision*recall) / (precision + recall - (precision*recall))
  f1 = (2*precision*recall)/(precision+recall)
  print("\tPoint perecentage:", point_percent)
  print("\tPrecision        :", precision)
  print("\tRecall           :", recall)
  print("\tiou              :", iou)
  print("\tF1               :", f1)

  # Visualize result
  for file_name in evaluation_data_file_names:
    visualization_data = pd.read_csv(get_dataset_path(file_name), index_col=0)
    visualization_features = visualization_data.drop(["target"] + remove_features, axis=1)
    predictions = model.predict(visualization_features)

    pcd = o3d.geometry.PointCloud()
    points = visualization_data[["x","y","z"]].to_numpy()
    pcd.points = o3d.utility.Vector3dVector(points)

    # TODO: use same colors as in roof cloud test images
    colors = np.zeros((predictions.shape[0], 3))
    colors += [0.6, 0.6, 0.6]
    colors[predictions < 0.5] = [0, 1, 0] # Color positive values as green
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

def test_feature_removal():
  worst_features = [] #"z", "x", "EdgeVoxels"]
  test_again = True

  # TODO: test removal of one and one feature, not just feature groups

  while test_again:
    training_data = pd.DataFrame()
    for file_name in training_data_file_names:
      data = pd.read_csv(get_dataset_path(file_name), index_col=0)
      training_data = pd.concat([training_data, data])

    # Loop through features and create a set with one removed at a time
    all_features = training_data.drop("target", axis=1)
    feature_list = all_features.columns.tolist()

    # features_names = feature_list
    features_names = ["x", "y", "z", "10", "kNNCentroidDistance", "LowerVoxels", "UpperVoxels", "AroundVoxels", 
                      "CovarianceEigenvalue_feature_0", "CovarianceEigenvalue_feature_1", "CovarianceEigenvalue_feature_2", 
                      "CovarianceEigenvalue_feature_3", "CovarianceEigenvalue_feature_4", "CovarianceEigenvalue_feature_5", 
                      "CovarianceEigenvalue_feature_6", "CovarianceEigenvalue_feature_7", "CovarianceEigenvalue_feature_8", 
                      "CovarianceEigenvalue_feature_9", "CovarianceEigenvalue_feature_10", "CovarianceEigenvalue_feature_11"] #, "EdgeVoxels"] # "NormalCluster",
    features_names = [f for f in features_names if f not in worst_features]

    # Remove every NormalCluster
    grouped_features = [[f for f in feature_list if f.startswith(feature_name)] for feature_name in features_names]
    # grouped_features = [[feature_name] for feature_name in features_names]

    drop_features = [f for feature_name in worst_features for f in feature_list if f.startswith(feature_name)]
    # drop_features = worst_features

    results = np.zeros((len(features_names) + 1, 5))

    def run_without_features(without_features):
      # Split in traning and testing splits
      features = training_data.drop(["target"] + without_features, axis=1)
      targets = training_data["target"]
      X_train, X_validation, y_train, y_validation = train_test_split(features, targets, train_size=0.8, random_state=1234)

      # Weight the edge / non-edge classes
      target_weight = calculate_weight(y_train)

      train_pool = Pool(
        data=X_train,
        label=y_train,
        weight=target_weight,
      )

      model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        random_seed=42,
        logging_level="Silent"
      )

      model.fit(train_pool, eval_set=(X_validation, y_validation), verbose=True)

      evaluation_data = pd.DataFrame()
      for file_name in evaluation_data_file_names:
        data = pd.read_csv(get_dataset_path(file_name), index_col=0)
        evaluation_data = pd.concat([evaluation_data, data])

      evaluation_features = evaluation_data.drop(["target"] + without_features, axis=1)
      targets = evaluation_data["target"].to_numpy()

      predictions = model.predict(evaluation_features)
      
      point_percent = np.sum(np.equal(predictions, targets))/predictions.shape[0]
      precision = precision_score(targets, predictions)
      recall = recall_score(targets, predictions)

      # Formulas from: https://tomkwok.com/posts/iou-vs-f1/
      iou = (precision*recall) / (precision + recall - (precision*recall))
      f1 = (2*precision*recall)/(precision+recall)
      
      res = []
      res.append(point_percent)
      res.append(precision)
      res.append(recall)
      res.append(iou)
      res.append(f1)

      return res
    

    for f_i, remove_feature in enumerate(tqdm(grouped_features)):
      res = run_without_features(drop_features + remove_feature)
      results[f_i] = res

    res = run_without_features(drop_features)
    results[len(features_names)] = res

    features_names.append("Only Worst")

    for i in range(len(features_names)):
      print("Removing:", features_names[i])
      print("\t\tpoint_percent:", results[i, 0])
      print("\t\tprecision    :", results[i, 1])
      print("\t\trecall       :", results[i, 2])
      print("\t\tiou          :", results[i, 3])
      print("\t\tF1           :", results[i, 4])

    best_point_percent_i = np.argmax(results[:,0])
    best_precision_i = np.argmax(results[:,1])
    best_recall_i = np.argmax(results[:,2])
    best_iou_i = np.argmax(results[:,3])
    best_f1_i = np.argmax(results[:,4])
    best_sum_i = np.argmax(np.sum(results, axis=1))

    print("\n\nBest to remove:")
    print("\tpoint_percent :", features_names[best_point_percent_i], "\twith", results[best_point_percent_i, 0])
    print("\tprecision     :", features_names[best_precision_i], "\twith", results[best_precision_i, 1])
    print("\trecall        :", features_names[best_recall_i], "\twith", results[best_recall_i, 2])
    print("\tiou           :", features_names[best_iou_i], "\twith", results[best_iou_i, 3])
    print("\tF1            :", features_names[best_f1_i], "\twith", results[best_f1_i, 4])
    print("\tsum           :", features_names[best_sum_i], "\twith", np.sum(results[best_sum_i]))

    if best_iou_i != len(features_names) - 1:
      print("\nAdding", features_names[best_iou_i], "to worst features\n\n\n")
      worst_features.append(features_names[best_iou_i])
      print("The worst features so fare are:", worst_features)
    else:
      test_again = False

  print("The worst features which should be removed are:", worst_features)

def test_feature():
  # FUNCTIONS
  def get_feature_index():
    for file_i, file in enumerate(feature_names):
      print("(" + str(file_i) + ")\t" + file)
    chosen_index = int(input("> "))
    while chosen_index < 0 or chosen_index >= len(feature_names):
      print("Must be between 0 and " + str(len(feature_names) - 1))
      chosen_index = int(input("> "))
    return chosen_index
  
  def get_cloud_file_name():
    for file_name_i, file_name in enumerate(test_file_names):
      print("(" + str(file_name_i) + ")\t" + file_name)
    print("(-1)\tAll clouds")
    chosen_index = int(input("> "))
    while chosen_index != -1 and (chosen_index < 0 or chosen_index >= len(test_file_names)):
      print("Must be between 0 and " + str(len(test_file_names) - 1))
      chosen_index = int(input("> "))
    return chosen_index

  def test_feature_on_cloud(feature_index, cloud_file_name):
    print("Testing out", all_features[feature_index][2], "on", cloud_file_name)
    cloud = read_roof_cloud(cloud_file_name)
    cloud, downsampling_factor = normalize_cloud(cloud)
    cloud, indices = remove_noise(cloud)
    state = all_features[feature_index][1](cloud, downsampling_factor)
    f = all_features[feature_index][0](state)
    f.run_test(all_features[feature_index][2], cloud_file_name)

  # PROGRAM
  print("Which feature do you want to test?:")
  feature_index = get_feature_index()

  print("Which cloud file do you want to test?:")
  cloud_index = get_cloud_file_name()

  if cloud_index == -1:
    for i in range(len(test_file_names)):
      test_feature_on_cloud(feature_index, test_file_names[i])
  else:
    test_feature_on_cloud(feature_index, test_file_names[cloud_index])

def plot_feature_figure():
  def get_feature_index(choices):
    for choice_i, choice in enumerate(choices):
      print("(" + str(choice_i) + ") " + str(choice))
    chosen_index = int(input("> "))

    while (chosen_index < 0 or chosen_index >= len(choices)):
      print("Must be between 0 and " + str(len(choices) - 1))
      chosen_index = int(input("> "))
    
    return chosen_index

  # Ask for which feature
  features_array = np.array(all_features)
  choices = features_array[:, 2]
  chosen_index = get_feature_index(choices)
  chosen_feature_name = choices[chosen_index]

  # Ask for which file to display
  project_folder = get_project_folder()
  results_feature_folder = 'edge_detection/results/feature'
  image_folder = os.path.join(project_folder, results_feature_folder, f"{chosen_feature_name}/images/32-1-510-215-53-test-1")
  
  # print('Feature name 1       :', features_array[chosen_index, 1].__name__)
  # print('Feature name 0       :', features_array[chosen_index, 0].__name__)
  # print('Feature class 0      :', features_array[chosen_index, 0].__class__)
  # print('Feature typeof 0     :', type(features_array[chosen_index, 0]))
  # print('Feature bases[0].name:', features_array[chosen_index, 0].__bases__[0].__name__)

  parent_classes = features_array[chosen_index, 0].__bases__
  is_multifeature = len(parent_classes) > 0 and parent_classes[0].__name__ == "ScalableMultiFeature"
  if is_multifeature:
    # Ask for which sub-feature
    subfeatures = sorted([f for f in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, f))])
    # print('Subfeatures:', subfeatures)
    subfeature_name = subfeatures[get_feature_index(subfeatures)]
    image_folder = os.path.join(image_folder, subfeature_name)

  # Display all the scales
  images = sorted([f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and f.startswith("Scale")])
  print('Images:', images)
  
  figsize = [16, 8]
  fig, ax = plt.subplots(nrows=int(np.ceil(len(images)/5)), ncols=5, figsize=figsize) #, figsize=figsize)
  for i, axi in enumerate(ax.flat):
    axi.axis('off')
    if i < len(images):
      img = mpimg.imread(os.path.join(image_folder, images[i]))
      axi.imshow(img)
      axi.set_title(images[i])
  
  plt.tight_layout()
  plt.subplots_adjust(wspace=0, hspace=0)
  plt.show()

  # Ask for which scales to choose
  print("Which scales do you want to display?\nEnter numbers with space between:")
  map_to_scales = lambda scales: [int(s) for s in scales.strip().split(" ")]
  check_scales = lambda scales: [s >= 0 and s < len(images) for s in scales]

  for choice_i, choice in enumerate(images):
    print("(" + str(choice_i) + ") " + str(choice)) 
  chosen_scales = map_to_scales(input("> "))

  while len(chosen_scales) <= 0 or not all(check_scales(chosen_scales)):
    if len(chosen_scales) <= 0:
      print("Must select at least one scale!")
    else:
      print("All scales must be between 0 and " + str(len(images) - 1))
    chosen_scales = map_to_scales(input("> "))

  # Display the selected features with legend
  chosen_images = sorted([images[s] for s in chosen_scales])

  figsize = [len(chosen_images)*6, 6]
  ratio = [5 if i < len(chosen_images) else 1 for i in range(len(chosen_images)+1)]
  fig, ax = plt.subplots(nrows=1, ncols=len(chosen_images)+1, figsize=figsize, gridspec_kw={'width_ratios': ratio})
  colormap = cm.get_cmap('rainbow')

  for i, axi in enumerate(ax.flat):
    axi.axis('off')
    if i < len(chosen_images):
      img = mpimg.imread(os.path.join(image_folder, chosen_images[i]))
      axi.imshow(img, interpolation='nearest')
      # axi.set_title(chosen_images[i])

  gradient = np.linspace(0, 1, 256)
  gradient = np.vstack((gradient, gradient))
  plot = ax[-1].pcolor(gradient, cmap=colormap)
  ax[-1].set_visible(False)
  ax[-1].set_position([0,0,0,0])
  fig.colorbar(plot)

  plt.tight_layout()
  plt.subplots_adjust(wspace=0, hspace=0)
  if is_multifeature:
    save_name = f"{chosen_feature_name}_{subfeature_name}_plots.png"
  else:
    save_name = f"{chosen_feature_name}_plots.png"
  plt.savefig(os.path.join(image_folder, save_name), bbox_inches=0)

def calculate_feature_performance():
  training_data = pd.DataFrame()
  for file_name in training_data_file_names:
    data = pd.read_csv(get_dataset_path(file_name), index_col=0)
    training_data = pd.concat([training_data, data])
  
  # idx0 = training_data.index.values[training_data["target"] == 0]
  # idx1 = training_data.index.values[training_data["target"] == 1]
  # len0 = len(idx0)  # 0000
  # len1 = len(idx1)  # 2500
  # print(f'len0: {len0}, len1: {len1}')

  # Define feature set and feature list
  all_features = training_data.drop("target", axis=1)
  feature_list = all_features.columns.tolist()
  targets = training_data["target"]
  # X_train, X_validation, y_train, y_validation = train_test_split(features, targets, train_size=0.8, random_state=1234)

  def calculate_iou(predictions, targets):    
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    if precision + recall < 0.00001:
      # Avoid zero-division
      #print('return 0: add')
      return 0
    return np.divide(precision*recall, precision + recall - (precision*recall))
  
  def optimize_iou(values, targets, f_min, f_max, rec_left):
    step = (f_max - f_min) / 10
    # best iou (iou, threshold)
    iou_max = (0, 0)
    for threshold in [f_min + step*i for i in range(11)]: # range(f_min, f_max+step, step):
      # Need to test > and <
      pred_under = np.where(values < threshold, 1, 0)
      pred_over = np.where(values >= threshold, 1, 0)
      targets_set = set(targets)
      # print('set pred_under', set(pred_under))
      # print('set pred_over ', set(pred_over))
      # print('set targets   ', set(targets))
      if len(targets_set - set(pred_under)) > 0:
        iou_under = 0
      else:
        iou_under = calculate_iou(pred_under, targets)

      if len(targets_set - set(pred_over)) > 0:
        iou_over = 0
      else:
        iou_over = calculate_iou(pred_over, targets)

      if iou_under > iou_max[0]:
        iou_max = (iou_under, threshold)
      
      if iou_over > iou_max[0]:
        iou_max = (iou_over, threshold)

    if rec_left > 0:
      new_f_min = iou_max[1]-step
      new_f_max = iou_max[1]+step
      return optimize_iou(values, targets, new_f_min, new_f_max, rec_left - 1)
    else:
      return iou_max


  one_targets = sum([1 if x == 1 else 0 for x in targets])
  zero_targets = sum([1 if x == 0 else 0 for x in targets])
  one_weight = zero_targets/(one_targets+zero_targets)
  zero_weight = one_targets/(one_targets+zero_targets)

  print(f'zero_weight: {zero_weight}, one_weight: {one_weight}')

  # test_values = np.random.randint(0, 2, targets.shape[0])
  test_values = np.random.choice([0, 1], size=targets.shape[0]) #, p=[.154, .846])
  test_targets = np.random.choice([0, 1], size=targets.shape[0], p=[.154, .846])
  print('random test IoU:', calculate_iou(test_values, test_targets))

  
  feature_performance = {}
  for feature_name in tqdm(feature_list):
    # print(f"\tCalculating for feature: {feature_name}")
    feature_values = all_features[feature_name].to_numpy()
    f_min = np.min(feature_values)
    f_max = np.max(feature_values)
    # pred_under = np.where(feature_values < f_min, 0, 1)
    # pred_over = np.where(feature_values >= f_min, 0, 1)
    # print('set pred_under', set(pred_under))
    # print('set pred_over ', set(pred_over))
    # print('set targets   ', set(targets))
    # print('diff:', len(set(targets) - set(pred_under)))
    iou, threshold = optimize_iou(feature_values, targets, f_min, f_max, 3)
    # print(f"\t\t Best IoU was {iou} with threshold {threshold}")
    feature_performance[feature_name] = iou
    # break
  
  feature_groups = ["kNNCentroidDistance", "LowerVoxels", "UpperVoxels", "AroundVoxels", "EdgeVoxels",
                     "CovarianceEigenvalue_feature_0", "CovarianceEigenvalue_feature_1", "CovarianceEigenvalue_feature_2", 
                     "CovarianceEigenvalue_feature_3", "CovarianceEigenvalue_feature_4", "CovarianceEigenvalue_feature_5", 
                     "CovarianceEigenvalue_feature_6", "CovarianceEigenvalue_feature_7", "CovarianceEigenvalue_feature_8", 
                     "CovarianceEigenvalue_feature_9", "CovarianceEigenvalue_feature_10"]

  for group in feature_groups:
    s = group
    for key, val, in sorted(feature_performance.items()):
      if group in key:
        s += f"\t&\t{(val*10):.4}"
    
    s += "\\\\"
    print("\\hline")
    print(s)
    

  # print("Feature performance:")
  # for key, val, in sorted(feature_performance.items()):
  #   print(f"{key: <42}: {val:.6}")
  
    

  
  

def main():
  main_menu_choice = get_main_menu_choice()
  while main_menu_choice != -1:
    # choices = ["Calculate features", "Classify points", "Calculate and classify", "Test feature removal", "Test feature"]
    if main_menu_choice == 0:
      print('Calculate features')
      calculate_features()
    elif main_menu_choice == 1:
      print('Classify points')
      classify_points()
    elif main_menu_choice == 2:
      print('Calculate and classify')
    elif main_menu_choice == 3:
      print('Test feature removal')
      test_feature_removal()
    elif main_menu_choice == 4:
      print('Test feature')
      test_feature()
    elif main_menu_choice == 5:
      print('Plot feature figure')
      plot_feature_figure()
    elif main_menu_choice == 6:
      print('Calculate feature performance')
      calculate_feature_performance()

    main_menu_choice = get_main_menu_choice()




if __name__ == "__main__":
  main()
