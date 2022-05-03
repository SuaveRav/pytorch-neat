import pandas as pd
import numpy as np
from twolayer import TwoLayerNet
import pickle
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def categorize_label(outputs_list):
    new_outputs = []
    for target in outputs_list:
        if target == 0.0:
            new_outputs.append(0)
        if target == .5:
            new_outputs.append(1)
        if target == 1.0:
            new_outputs.append(2)
    return new_outputs

scenarios = ['KQK', 'KPK', 'KRK', 'KRKN']
results = {'KQK' : {},
'KPK' : {},
'KRK' : {},
'KRKN' : {}
}

for scenario in scenarios:
    np_rng = np.random.default_rng(13)
    print("Scenario: {}".format(scenario))
    with open("./data/{}/indices".format(scenario), "rb") as f:
        data = pickle.load(f)
    np_rng.shuffle(data)

    # train data
    inputs_list = []
    outputs_list = []
    data_proportion = 2000
    data_train = data[:data_proportion]
    for xy in data_train:
        # xy[0] should already be a numpy array
        inputs_list.append(xy[0] / 8)
        outputs_list.append(xy[1])

    # test data 
    inputs_list_test = []
    outputs_list_test = []
    data_test = data[2000:4000]
    losses = []

    for xy in data_test:
        # xy[0] should already be a numpy array
        inputs_list_test.append(xy[0] / 8)
        outputs_list_test.append(xy[1])

    # test_results = pd.read_csv("./images/17/test_results.csv")
    test_results = pd.read_csv("./{}_test_results/test_results.csv".format(scenario))
    # test_results = pd.read_csv("./KRK_test_results/test_results.csv")

    print("Average Loss: {}".format(test_results['loss'].mean()))
    average_loss = test_results['loss'].mean()
    test_results["label"] = outputs_list_test

    def win_lose_or_draw(prediction):
        diffs = np.array([np.abs(1 - prediction), np.abs(.5 - prediction), np.abs(0 - prediction)])
        return np.argmin(diffs)

    correct_predictions = 0

    actual_predictions = []
    for index, row in test_results.iterrows():
        prediction_idx = win_lose_or_draw(row['prediction'])
        if prediction_idx == 0:
            prediction = 1.0
        if prediction_idx == 1:
            prediction = .5
        if prediction_idx == 2:
            prediction = 0.0
        actual_predictions.append(prediction)
        truth_label = row['label']

        if prediction == truth_label:
            correct_predictions += 1

    print("Overall WANN Accuracy: {}".format(correct_predictions / 2000))
    print("WANN Fitness: {}".format(2000 * (1 - average_loss)))
    print("WANN Confusion Matrix: \n{}".format(confusion_matrix(categorize_label(outputs_list_test), categorize_label(actual_predictions))))

    results[scenario]['Accuracy WANN'] = correct_predictions / 2000
    results[scenario]['Confusion Matrix WANN'] = confusion_matrix(categorize_label(outputs_list_test), categorize_label(actual_predictions)).tolist()

    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(8, 4, 1), random_state=1)

    clf.fit(inputs_list, categorize_label(outputs_list))
    weights = clf.coefs_
    print(sum(len(row) for row in weights))
    predictions = clf.predict(inputs_list_test)
    print("Sklearn Network Accuracy: {}".format(accuracy_score(categorize_label(outputs_list_test), predictions)))
    print("Sklearn Confusion Matrix: \n{}".format(confusion_matrix(categorize_label(outputs_list_test), predictions)))
    print("")
    results[scenario]['Accuracy Sklearn NN'] = accuracy_score(categorize_label(outputs_list_test), predictions)
    results[scenario]['Confusion Matrix Sklearn NN'] = confusion_matrix(categorize_label(outputs_list_test), predictions).tolist()

# print(results)
with open('./results.json', 'w') as convert_file:
    convert_file.write(json.dumps(results))

    