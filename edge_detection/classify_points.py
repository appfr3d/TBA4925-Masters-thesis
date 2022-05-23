import os
import numpy as np
import pandas as pd
import open3d as o3d
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

np.set_printoptions(precision=4)

def get_dataset_path(file_name):
  current_folder = os.path.dirname(os.path.abspath(__file__))
  project_folder = os.path.dirname(current_folder)
  return os.path.join(project_folder, "data/point_clouds/classified_roofs/calculated_values", file_name)

training_data_file_names = ["32-1-510-215-53-test-1.csv", "32-1-510-215-53-test-2.csv", "32-1-510-215-53-test-3.csv", 
                            "32-1-510-215-53-test-4.csv", "32-1-510-215-53-test-5.csv", "32-1-510-215-53-test-6.csv",
                            "32-1-510-215-53-test-9.csv", "32-1-510-215-53-test-12.csv", "32-1-510-215-53-test-14.csv",
                            "32-1-510-215-53-test-10.csv", "32-1-510-215-53-test-large-1.csv"]

evaluation_data_file_names = ["32-1-510-215-53-test-11.csv", "32-1-510-215-53-test-8.csv",
                              "32-1-510-215-53-test-13.csv", "32-1-510-215-53-test-7.csv"]

def calculate_weight(targets):
  one_targets = sum([1 if x == 1 else 0 for x in targets])
  zero_targets = sum([1 if x == 0 else 0 for x in targets])
  one_weight = zero_targets/(one_targets+zero_targets)
  zero_weight = one_targets/(one_targets+zero_targets)
  return [one_weight if target == 1 else zero_weight for target in targets]


def run_classification():
  training_data = pd.DataFrame()
  for file_name in training_data_file_names:
    data = pd.read_csv(get_dataset_path(file_name), index_col=0)
    training_data = pd.concat([training_data, data])

  # Remove feature groups
  all_features = training_data.drop("target", axis=1)
  feature_list = all_features.columns.tolist()

  # removing "kNNCentroidDistance" gives better metric results, but not visually better results...
  # "kNNCentroidDistance", "NormalCluster", "EdgeVoxels", "AroundVoxels", "UpperVoxels", "LowerVoxels"
  remove_group_features = ["LowerVoxels"]
  remove_single_features = [] #"x", "z"]

  ''' 
  Whitout covariance_eigenvalue scaling
  Evaluation:
	Point perecentage: 0.9130725509862923
	Precision        : 0.9608472400513479
	Recall           : 0.9279274202637214
	iou              : 0.8941195634468154
	F1               : 0.9441004471964224

  With covariance_eigenvalue scaling
  Evaluation:
	Point perecentage: 0.9089044912515324
	Precision        : 0.9743527776938646
	Recall           : 0.9087681731094331
	iou              : 0.8875374920888254
	F1               : 0.9404183978424084
  '''

  remove_features = []

  # Remove groups
  for feature_group_name in remove_group_features:
    remove_features += [f for f in feature_list if f.startswith(feature_group_name)]

  # Remove singles
  remove_features += [f for f in remove_single_features if f not in remove_features]

  print("Classifying without features:", remove_features)

  features = training_data.drop(["target"] + remove_features, axis=1)
  targets = training_data["target"]


  # Testing with categorical features made it perform worse
  # categorical_features = [f"LowerVoxels_{i}" for i in range(8)] + [f"UpperVoxels_{i}" for i in range(8)]# [f"AroundVoxels_{i}" for i in range(8)]
  # actual_cat_features = []
  # for cat in categorical_features:
  #   if not cat in remove_features:
  #     # actual_cat_features.append(cat)
  #     # features[cat] = features[cat].apply(int)
  #     print(features[cat].dtype)

  # Split in traning and testing splits
  X_train, X_validation, y_train, y_validation = train_test_split(features, targets, train_size=0.8, random_state=1234)

  # Weight the edge /non-edge classes
  target_weight = calculate_weight(y_train)

  train_pool = Pool(
    # cat_features=actual_cat_features,
    data=X_train,
    label=y_train,
    weight=target_weight,
  )

  model = CatBoostClassifier(
    iterations=500,
    ## early_stopping_rounds=100,
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
  pd.set_option('display.max_rows', 107)
  print(feature_importance)

  evaluation_data = pd.DataFrame()
  for file_name in evaluation_data_file_names:
    data = pd.read_csv(get_dataset_path(file_name), index_col=0)
    evaluation_data = pd.concat([evaluation_data, data])

  evaluation_features = evaluation_data.drop(["target"] + remove_features, axis=1)
  evaluation_targets = evaluation_data["target"].to_numpy()

  # for cat in actual_cat_features:
  #   evaluation_features[cat] = evaluation_features[cat].apply(int)

  predictions = model.predict(evaluation_features)

  # Evaluation metrics
  print("Evaluation:")

  # true_positives = np.bitwise_and(np.array(predictions, dtype=bool), np.array(evaluation_targets, dtype=bool))
  # print("\tTP perecentage   :", np.sum(true_positives)/np.sum(evaluation_targets))

  point_percent = np.sum(np.equal(predictions, evaluation_targets))/predictions.shape[0]
  print("\tPoint perecentage:", point_percent)

  # positives = np.sum(predictions)
  # negatives = predictions.shape[0] - positives

  precision = precision_score(evaluation_targets, predictions)
  recall = recall_score(evaluation_targets, predictions)

  # Formulas from: https://tomkwok.com/posts/iou-vs-f1/
  iou = (precision*recall) / (precision + recall - (precision*recall))
  f1 = (2*precision*recall)/(precision+recall)
  print("\tPrecision        :", precision)
  print("\tRecall           :", recall)
  print("\tiou              :", iou)
  print("\tF1               :", f1)

  # Visualize result
  for file_name in evaluation_data_file_names:
    visualization_data = pd.read_csv(get_dataset_path(file_name), index_col=0)
    visualization_features = visualization_data.drop(["target"] + remove_features, axis=1)

    # for cat in actual_cat_features:
    #   visualization_features[cat] = visualization_features[cat].apply(int)

    predictions = model.predict(visualization_features)
    pcd = o3d.geometry.PointCloud()
    points = visualization_data[["x","y","z"]].to_numpy()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.zeros((predictions.shape[0], 3))
    colors += [0.6, 0.6, 0.6]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])

    colors[predictions < 0.5] = [0, 1, 0] # Color positive values as green

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def test_feature_removal():
  worst_features = [] #"z", "x", "EdgeVoxels"]
  test_again = True

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
                       "CovarianceEigenvalue_feature_6", "CovarianceEigenvalue_feature_7", ] #, "EdgeVoxels"] # "NormalCluster",
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

      # Weight the edge /non-edge classes
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
      evaluation_targets = evaluation_data["target"].to_numpy()

      predictions = model.predict(evaluation_features)
      
      point_percent = np.sum(np.equal(predictions, evaluation_targets))/predictions.shape[0]
      precision = precision_score(evaluation_targets, predictions)
      recall = recall_score(evaluation_targets, predictions)

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

def class_imbalance():
  training_data = pd.DataFrame()
  for file_name in training_data_file_names:
    data = pd.read_csv(get_dataset_path(file_name), index_col=0)
    training_data = pd.concat([training_data, data])

  targets = training_data["target"]
  # Weight the edge /non-edge classes
  one_targets = sum([1 if x == 1 else 0 for x in targets])
  zero_targets = sum([1 if x == 0 else 0 for x in targets])
  one_weight = zero_targets/(one_targets+zero_targets)
  zero_weight = one_targets/(one_targets+zero_targets)
  print(f"one_weight: {one_weight}\nzero_weight: {zero_weight}")

def main():
  # test_feature_removal()
  run_classification()
  # class_imbalance()

  # Removing whole  features by sum     : best to remove kNNCentroidDistance and y
  # Removing single features by sum     : best to remove kNNCentroidDistance and LowerVoxels_2
  # Removing single features by recall  : best to remove kNNCentroidDistance and NormalCluster_7


if __name__ == "__main__":
  main()



'''
After removing kNNCentroidDistance:

Removing: x
		point_percent: 0.9024295432458698
		precision    : 0.9626498422712934
		recall       : 0.915077365958978
Removing: y
		point_percent: 0.9077745383867832
		precision    : 0.9666498231430015
		recall       : 0.917836152093079
Removing: z
		point_percent: 0.9051506316812439
		precision    : 0.9589724404539219
		recall       : 0.9223941465755068
Removing: LowerVoxels
		point_percent: 0.9042759961127308
		precision    : 0.9634392334846192
		recall       : 0.9166366798608612
Removing: UpperVoxels
		point_percent: 0.9064139941690962
		precision    : 0.9622617853560682
		recall       : 0.9205949382271801
Removing: AroundVoxels
		point_percent: 0.9027210884353741
		precision    : 0.9603413654618473
		recall       : 0.917836152093079
Removing: NormalCluster
		point_percent: 0.9045675413022352
		precision    : 0.9596300462442194
		recall       : 0.9209547798968454
Removing: CovarianceEigenvalue
		point_percent: 0.8806608357628766
		precision    : 0.960248608053865
		recall       : 0.8895286074127384
Removing: EdgeVoxels
		point_percent: 0.8984450923226434
		precision    : 0.9590783178040796
		recall       : 0.9136379992803166
Removing: Only Worst
		point_percent: 0.9069970845481049
		precision    : 0.9622901528438987
		recall       : 0.9213146215665108


Best to remove:
	point_percent : y 	with 0.9077745383867832
	precision     : y 	with 0.9666498231430015
	recall        : z 	with 0.9223941465755068
	sum           : y 	with 2.7922605136228635
'''