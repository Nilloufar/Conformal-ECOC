# %%
from matplotlib import pyplot as plt
# %%
import numpy as np
from sklearn.svm import SVC
import itertools
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import glob
import os


# %%

class binary_classifier:
    def __init__(self, ternary_code, classifier, q_hat=None):
        self.ternary_code = ternary_code
        self.model = classifier
        self.q_hat = q_hat

    def set_q_hat(self, q_hat):
        self.q_hat = q_hat


class ECOC:
    def __init__(self, n_classes, data_dic, base_classifier=SVC(), with_conformal_prediction=False, ecoc_type='ova'):
        self.base_classifier = base_classifier
        self.classifiers = []
        self.n_classes = n_classes
        self.ecoc_type = ecoc_type
        if self.ecoc_type == "ovo":
            self.n_classifiers = n_classes * (n_classes - 1) // 2
        elif self.ecoc_type == "random_sparse":
            self.n_classifiers = int(15 * np.log2(n_classes))
        elif self.ecoc_type == "random_dense":
            self.n_classifiers = int(10 * np.log2(n_classes))
        elif self.ecoc_type == "ova":
            self.n_classifiers = n_classes
        self.ecoc_matrix = self.generate_ecoc()
        self.with_conformal_prediction = with_conformal_prediction
        self.q_hat_list = []
        self.data_dic = data_dic

    def train(self):
        X, y = self.data_dic['train']
        for i in range(self.n_classifiers):
            ternary_code = self.ecoc_matrix[:, i]

            filtered_X, filtered_y = self.filter_data(ternary_code, X, y)

            # print("Training classifier {} with labels 0: {} and 1 :{}".format(i, np.where(ternary_code==-1), np.where(ternary_code==1)))
            classifier = clone(self.base_classifier)
            if filtered_X.shape[0] == 0:
                print (ternary_code)
                exit()
            classifier.fit(filtered_X, filtered_y)
            q_hat = None

            if self.with_conformal_prediction:
                q_hat = self.find_q_hat(ternary_code, classifier)

            self.classifiers.append((ternary_code, classifier, q_hat))

    def filter_data(self, ternary_code, X, y):
        class_0 = np.where(ternary_code == -1)[0]  # label as 0
        class_1 = np.where(ternary_code == 1)[0]  # label as 1
        mask = np.isin(y, np.concatenate((class_0, class_1)))
        filtered_y = y[mask]
        filtered_y = np.where(np.isin(filtered_y, class_0), 0, np.where(np.isin(filtered_y, class_1), 1, filtered_y))
        filtered_X = X[mask, :]
        return filtered_X, filtered_y

    def get_unused_data(self, ternary_code, X, y):
        class_0 = np.where(ternary_code == -1)[0]  # label as 0
        class_1 = np.where(ternary_code == 1)[0]  # label as 1
        mask = ~np.isin(y, np.concatenate((class_0, class_1)))
        filtered_y = y[mask]
        filtered_X = X[mask, :]
        return filtered_X, filtered_y

    def find_q_hat(self, ternary_code, classifier):

        X, y = self.data_dic['val']
        seen_X, seen_y = self.filter_data(ternary_code, X, y)
        if seen_X.shape[0] == 0:
            return 0.5
        else:
            prediction_probs = classifier.predict_proba(seen_X)
            score_of_true_class= np.array([prediction_probs[i][seen_y[i]] for i in range(len(seen_y))])
            q_hat = np.quantile(score_of_true_class, 0.1, method='lower')

        # prediction = classifier.predict(X)
        best_performance = 1 # accuracy of rejected samples
        best_q_hat=0.5
        # for alpha in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1]:
        #     q_hat = np.quantile(score_of_true_class,alpha, method='lower')
        #     prediction_probs=classifier.predict_proba(X)
        #     count_greater_than_q_hat = np.sum(prediction_probs > q_hat, axis=1)
        #     reject_index = np.where(count_greater_than_q_hat != 1)[0]
        #     if len(reject_index) > 0:
        #         new_performance = np.sum(prediction[reject_index] == seen_y[reject_index]) / len(seen_y[reject_index])
        #         if new_performance < best_performance:
        #             best_performance = new_performance
        #             best_q_hat = q_hat
        # q_hat = best_q_hat
        return q_hat

    def predict(self, X):
        predictions = []
        weights= []
        for i in range(len(self.classifiers)):
            binary_predictions,weight = self.predict_outputs_with_conformal_weights(X, self.classifiers[i])
            predictions.append(binary_predictions)
            weights.append(weight)
        return self.decode_labels(np.column_stack(predictions),np.column_stack(weights))

    def get_ternary_labels (self, y, ternary_code):
        ternary_labels = np.zeros(len(y))
        class_0_labels = np.where(ternary_code == -1)[0]
        class_1_labels = np.where(ternary_code == 1)[0]
        ternary_labels[np.isin(y, class_0_labels)] = -1
        ternary_labels[np.isin(y, class_1_labels)] = 1

        return ternary_labels

    def predict_outputs_with_conformal_weights(self, X, classifier):
        ternary_code, classifier, q_hat = classifier
        binary_predictions = classifier.predict(X)
        binary_predictions = np.where(binary_predictions == 0, -1, binary_predictions)
        count_greater_than_q_hat = np.ones(X.shape[0])
        if self.with_conformal_prediction:
            q_hat = q_hat
            prediction_probs = classifier.predict_proba(X)
            count_greater_than_q_hat = np.sum(prediction_probs>q_hat, axis=1)

        return binary_predictions,count_greater_than_q_hat

    def generate_ecoc(self):
        if self.ecoc_type == 'ovo':
            ecoc_matrix = np.zeros((self.n_classes, self.n_classifiers))
            class_combinations = list(itertools.combinations(range(self.n_classes), 2))
            for i, combination in enumerate(class_combinations):
                class_i, class_j = combination
                ecoc_matrix[class_i, i] = 1
                ecoc_matrix[class_j, i] = -1
        elif self.ecoc_type == 'random_sparse':
            elements = [0, 1, -1]
            probabilities = [0.5, 0.25, 0.25]
            best_ecoc_matrix = None
            best_score = -np.inf
            for i in range(2000):
                ecoc_matrix = np.random.choice(elements, size=(self.n_classes, self.n_classifiers), p=probabilities)
                ecoc_matrix=self._check_ecoc_validity(ecoc_matrix)
                score = np.mean(distance.cdist(ecoc_matrix, ecoc_matrix, metric='hamming'))
                if score > best_score:
                    best_score = score
                    best_ecoc_matrix = ecoc_matrix

            ecoc_matrix = best_ecoc_matrix

        elif self.ecoc_type == 'ova':
            ecoc_matrix = np.ones((self.n_classes, self.n_classifiers))
            for i in range(self.n_classes):
                ecoc_matrix[i, i] = -1
        elif self.ecoc_type == 'random_dense':
            elements = [1, -1]
            probabilities = [0.3,0.7]
            best_ecoc_matrix = None
            best_score = -np.inf
            for i in range(5000):
                ecoc_matrix = np.random.choice(elements, size=(self.n_classes, self.n_classifiers), p=probabilities)
                ecoc_matrix = self._check_ecoc_validity(ecoc_matrix)
                score = self._get_ecoc_score(ecoc_matrix)
                if score > best_score:
                    best_score = score
                    best_ecoc_matrix = ecoc_matrix

            ecoc_matrix = best_ecoc_matrix




        else:
            raise ValueError('ecoc_type should be ovo ,ova, random_sparse or random_dense')
        return ecoc_matrix

    def _get_ecoc_score(self,ecoc_matrix):
        distances=[]
        for i in range(ecoc_matrix.shape[0]-1):
            for j in range(i+1,ecoc_matrix.shape[0]):
                distances.append(distance.hamming(ecoc_matrix[i,:],ecoc_matrix[j,:]))
        return np.mean(distances)


    def _check_ecoc_validity(self,ecoc_matrix):
        #  check at least one -1 , one 1 and one 0 in each column
        for i in range(ecoc_matrix.shape[1]):
            column = ecoc_matrix[:, i].copy()
            if self.ecoc_type == 'random_sparse':
                if not (-1 in column) :
                    # randomly change two 0 to 1 and -1
                    zero_indices = np.where(column == 0)[0]
                    if len(zero_indices) > 1:
                        random_index = np.random.choice(zero_indices, size=1, replace=False)
                        column[random_index] = -1
                    else:
                        one_indices = np.where(column == 1)[0]
                        random_index = np.random.choice(one_indices, size=1, replace=False)
                        column[random_index] = -1

                if not (1 in column):
                    zero_indices = np.where(column == 0)[0]
                    if len(zero_indices) > 1:
                        random_index = np.random.choice(zero_indices, size=1, replace=False)
                        column[random_index] = 1
                    else:
                        one_indices = np.where(column == -1)[0]
                        random_index = np.random.choice(one_indices, size=1, replace=False)
                        column[random_index] = 1

                if not (0 in column) :
                    # randomly change one 1 to 0 or -1
                    one_indices = np.where(column == 1)[0]
                    minus_one_indices = np.where(column == -1)[0]
                    if len(one_indices) > 1:
                        random_index = np.random.choice(one_indices, size=1, replace=False)
                        column[random_index] = 0
                    elif len(minus_one_indices) > 1:
                        random_index = np.random.choice(minus_one_indices, size=1, replace=False)
                        column[random_index] = 0
            else:
                if not(1 in column):
                    indices = np.where(column == -1)[0]
                    random_index = np.random.choice(indices, size=1, replace=False)
                    column[random_index] = 1
                if not(-1 in column):
                    indices = np.where(column == 1)[0]
                    random_index = np.random.choice(indices, size=1, replace=False)
                    column[random_index] = -1

            ecoc_matrix[:, i] = column
        return ecoc_matrix


    def decode_labels(self, predictions,weights):
        # print ("decoding predictions: ", predictions)
        # print(self.ecoc_matrix)
        min_distances = []
        for pred_row,weight in zip(predictions,weights):
            involve_classifiers_index = np.where(weight ==1)[0]
            if len(involve_classifiers_index)==0:
                involve_classifiers_index=list(np.arange(self.n_classifiers))
            pred_row=pred_row[involve_classifiers_index]
            ecoc_subset= self.ecoc_matrix[:, involve_classifiers_index]
            hamming_distances = distance.cdist([pred_row], ecoc_subset, metric='hamming')
            min_distance_index = np.argmin(hamming_distances)
            min_distances.append(min_distance_index)
        return min_distances


if __name__ == '__main__':
    print("{:<20}            {:<20}          {:<20}     {:<20}    {:<10}".format("data", "ECOC accuracy",
                                                                                 "conformed ECOC accuracy",
                                                                                 "difference", "number of classes"))
    # np.random.seed(42)

    directory = 'data/'
    # Pattern to match CSV files
    file_pattern = '*.csv'
    # Iterate over CSV files in the directory
    for data_path in glob.glob(os.path.join(directory, file_pattern)):
        print(data_path)
        # data_path="data/shuttle.csv"
        if data_path in ['data/soybean.csv','data/zoo.csv',"data/satimage.csv"]:
            continue
        df = pd.read_csv(data_path, header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

        n_classes = len(np.unique(y))

        data_dic = {"train": (X_train, y_train), "test": (X_test, y_test), "val": (X_val, y_val)}
        # estimator= LogisticRegression(max_iter=3000,random_state=42)
        estimator = SVC( probability=True,random_state=42)
        ecoc = ECOC(n_classes, data_dic, estimator,
                    with_conformal_prediction=True, ecoc_type="random_dense")

        ecoc.train()
        conformed_ecoc_predictions = ecoc.predict(X_test)
        ecoc.with_conformal_prediction = False
        ecoc_predictions = ecoc.predict(X_test)
        ecoc_accuracy = np.sum(ecoc_predictions == y_test) / len(y_test) * 100
        conformed_ecoc_accuracy = np.sum(conformed_ecoc_predictions == y_test) / len(y_test) * 100
        print("{:<20}            {:<20}          {:<20}     {:<20}    {:<10}".format(data_path, ecoc_accuracy,
                                                                                     conformed_ecoc_accuracy,
                                                                                     conformed_ecoc_accuracy - ecoc_accuracy,
                                                                                     n_classes))




# %%
# for i in range(len(ecoc.classifiers)):
#     ternary_code, classifier, q_hat =ecoc.classifiers[i]
#     X_test, y_test = data_dic["test"]
#     ecoc.with_conformal_prediction = True
#     binary_predictions = ecoc.predict_ternary_outputs(X_test, ecoc.classifiers[i])
#     correct_labels=ecoc.get_ternary_labels(y_test,ternary_code)
#     accuracy = np.sum(binary_predictions == correct_labels) / len(correct_labels) * 100
#     print("conformal  ecoc accuracy:",accuracy)
#
#     ecoc.with_conformal_prediction=False
#     binary_predictions = ecoc.predict_ternary_outputs(X_test, ecoc.classifiers[i])
#     correct_labels = ecoc.get_ternary_labels(y_test, ternary_code)
#     accuracy = np.sum(binary_predictions == correct_labels) / len(correct_labels) * 100
#     print("ecoc accuracy:", accuracy)
#
#     print("------------------")
