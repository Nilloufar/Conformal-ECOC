#%%
from matplotlib import pyplot as plt
#%%
import numpy as np
from sklearn.svm import SVC
import itertools
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
#%%


class TernaryECOC:
    def __init__(self, n_classes, base_classifier=SVC(), n_classifiers=5,with_conformal_prediction=False):
        self.base_classifier = base_classifier
        self.classifiers = []
        self.num_classifiers = n_classifiers
        self.n_classes = n_classes
        if n_classifiers != n_classes * (n_classes - 1) / 2:
            raise ValueError("Number of classifiers must be equal to n_classes*(n_classes-1)/2")
        self.n_classifiers=n_classifiers
        self.ecoc_matrix= self.generate_ovo_codes()
        self.with_conformal_prediction=with_conformal_prediction
        self.q_hat_list=[]


    def train(self, X, y):
        for i in range(self.n_classifiers):
            ternary_code = self.ecoc_matrix[:, i]

            class_0 = np.where(ternary_code == -1)[0] # label as 0
            class_1 = np.where(ternary_code == 1)[0] # label as 1

            mask = np.isin(y, np.concatenate((class_0, class_1)))

            filtered_y=y[mask]
            filtered_y=np.where(np.isin(filtered_y, class_0), 0, np.where(np.isin(filtered_y, class_1), 1,filtered_y))
            filtered_X = X[mask,:]

            # print("Training classifier {} with labels 0: {} and 1 :{}".format(i, np.where(ternary_code==-1), np.where(ternary_code==1)))
            classifier = clone(self.base_classifier)
            classifier.fit(filtered_X, filtered_y)
            q_hat=None
            if self.with_conformal_prediction:
                q_hat=self.find_q_hat(classifier,filtered_X,filtered_y)

            self.classifiers.append((ternary_code, classifier,q_hat))


    def adjust_q_hat(self,calibration_X,calibration_y):
        for classifier in self.n_classifiers:
            q_hat=self.find_q_hat(classifier,calibration_X,calibration_y)
            self.q_hat_list.append(q_hat)





    def find_q_hat(self,classifier,X,y):
        prediction_probs=classifier.predict_proba(X)
        score_of_correct_class=[prediction_probs[i][y[i]] for i in range(len(y))]
        score_of_notcorrect_class=[prediction_probs[i][np.absolute(1-y[i])] for i in range(len(y))]
        difference=np.array(score_of_correct_class)-np.array(score_of_notcorrect_class)

        q_hat=np.quantile(difference,0.1)
        # q_hat=np.quantile(score_of_notcorrect_class,0.1)

        return q_hat


    def predict(self, X):
        predictions = []
        for i in range(len(self.classifiers)):
            ternary_code, classifier,q_hat = self.classifiers[i]
            binary_predictions = classifier.predict(X)
            binary_predictions = np.where(binary_predictions == 0, -1, binary_predictions)
            if self.with_conformal_prediction:
                prediction_probs=classifier.predict_proba(X)
                score_of_max_class= np.max(prediction_probs,axis=1)
                score_of_min_class= np.min(prediction_probs,axis=1)
                difference=np.array(score_of_max_class)-np.array(score_of_min_class)
                binary_predictions=np.where(difference<q_hat,0,binary_predictions)
            # print("Predictions for classifier {} class{}and {} are {}".format(i,class_combinations[i][0] ,class_combinations[i][1],np.unique(binary_predictions)))
            predictions.append(binary_predictions)
        return self.decode_labels(np.column_stack(predictions))

    def generate_ovo_codes(self):
        ecoc_matrix = np.zeros((self.n_classes, self.n_classifiers), dtype=int)
        class_combinations = list(itertools.combinations(range(self.n_classes), 2))
        for i, combination in enumerate(class_combinations):
            class_i, class_j = combination
            ecoc_matrix[class_i,i] = 1
            ecoc_matrix[class_j,i] = -1
        return ecoc_matrix

    def decode_labels(self, predictions):
        # print ("decoding predictions: ", predictions)
        # print(self.ecoc_matrix)
        min_distances = []
        for pred_row in predictions:
            hamming_distances = distance.cdist([pred_row], self.ecoc_matrix, metric='hamming')
            min_distance_index = np.argmin(hamming_distances)
            min_distances.append(min_distance_index)
        return min_distances


# # Example usage
# X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
# y_train = np.array([0, 1, 2, 1, 2, 0])
# X_test = np.array([[13, 14], [15, 16], [17, 18]])

# ecoc = TernaryECOC(3,LogisticRegression(),  3)
# ecoc.train(X_train, y_train)
# predictions = ecoc.predict(X_test)
# print(predictions)



