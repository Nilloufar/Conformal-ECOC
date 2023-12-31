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


class TernaryECOC:
    def __init__(self, n_classes, data_dic, base_classifier=SVC(), with_conformal_prediction=False, ecoc_type='ovo',decoding_method='hamming'):
        self.base_classifier = base_classifier
        self.classifiers = []
        self.n_classes = n_classes
        self.ecoc_type = ecoc_type
        if self.ecoc_type == "ovo":
            self.n_classifiers = n_classes * (n_classes - 1) // 2
        elif self.ecoc_type == "random_sparse":
            self.n_classifiers = int(15 * np.log2(n_classes))
        self.ecoc_matrix = self.generate_ecoc()
        self.with_conformal_prediction = with_conformal_prediction
        self.q_hat_list = []
        self.data_dic = data_dic
        self.decoding_method=decoding_method

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
        unseen_X, unseen_y = self.get_unused_data(ternary_code, X, y)
        if seen_X.shape[0] == 0:
            q_min, q_max = 0,100
            difference =[100]
            return (difference, np.inf,0,0)
        else:
            prediction_probs = classifier.predict_proba(seen_X)
            score_of_max_class = np.max(prediction_probs, axis=1)
            score_of_min_class = np.min(prediction_probs, axis=1)
            difference = np.array(score_of_max_class) - np.array(score_of_min_class)
            q_min, q_max = np.percentile(difference, 5), np.percentile(difference, 95)


        prediction_probs = classifier.predict_proba(unseen_X)
        score_of_max_class = np.max(prediction_probs, axis=1)
        score_of_min_class = np.min(prediction_probs, axis=1)
        difference2 = np.array(score_of_max_class) - np.array(score_of_min_class)

        # plt.hist(difference, bins=100, alpha=0.5, label='seen')
        # plt.hist(difference2, bins=100, alpha=0.5, label='unseen')
        # plt.legend(loc='upper right')
        # plt.show()

        q_min, q_max = np.min(difference), np.max(difference2)



        return (np.mean(difference), np.mean(difference2), np.abs(np.std(difference)),abs(np.mean(difference)- np.mean(difference2)))

    def predict(self, X):
        predictions = []
        for i in range(len(self.classifiers)):
            binary_predictions = self.predict_ternary_outputs(X, self.classifiers[i])
            predictions.append(binary_predictions)
        return self.decode_labels(np.column_stack(predictions))

    def get_ternary_labels (self, y, ternary_code):
        ternary_labels = np.zeros(len(y))
        class_0_labels = np.where(ternary_code == -1)[0]
        class_1_labels = np.where(ternary_code == 1)[0]
        ternary_labels[np.isin(y, class_0_labels)] = -1
        ternary_labels[np.isin(y, class_1_labels)] = 1

        return ternary_labels

    def predict_ternary_outputs(self, X, classifier):
        ternary_code, classifier, q_hat = classifier
        binary_predictions = classifier.predict(X)
        binary_predictions = np.where(binary_predictions == 0, -1, binary_predictions)
        true_ternary_labesl = self.get_ternary_labels(y_test, ternary_code)

        if self.with_conformal_prediction:
            q_min, q_max,q_std,diff = q_hat
            prediction_probs = classifier.predict_proba(X)
            score_of_max_class = np.max(prediction_probs, axis=1)
            score_of_min_class = np.min(prediction_probs, axis=1)
            difference = np.array(score_of_max_class) - np.array(score_of_min_class)
            entropy_of_max_class = [-score * np.log(score) for score in score_of_max_class]
            # binary_predictions = np.where((difference < q_min) | (difference > q_max), 0,binary_predictions)
            ecoc_error =np.sum(binary_predictions != true_ternary_labesl)
            if diff > 0:
                for i in range(len(difference)):
                    if abs(difference[i] - q_max) -abs(difference[i] - q_min) < 0.1:
                        binary_predictions[i] = 0

            correct_change=0
            incorrect_change=0
            wrong_prediction=0
            for i in range(len(binary_predictions)):
                if binary_predictions[i] == 0 and true_ternary_labesl[i] == 0:
                    correct_change+=1
                elif binary_predictions[i] == 0 and true_ternary_labesl[i] != 0:
                    incorrect_change+=1
                if binary_predictions[i] != 0 and true_ternary_labesl[i] != 0 & binary_predictions[i] != true_ternary_labesl[i]:
                    wrong_prediction+=1

            min_error = np.sum(true_ternary_labesl == 0)
            error = np.sum(binary_predictions != true_ternary_labesl)
            # print("correct change: ", correct_change,"incorrect_change:",incorrect_change, "min error ecoc: ", min_error, "error conformal: ", error, "ecoc_error: ", ecoc_error)
        return binary_predictions

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
                ecoc_matrix = self._check_ecoc_validity(ecoc_matrix)
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
            probabilities = [0.3, 0.7]
            best_ecoc_matrix = None
            best_score = -np.inf
            for i in range(5000):
                ecoc_matrix = np.random.choice(elements, size=(self.n_classes, self.n_classifiers), p=probabilities)
                ecoc_matrix = self._check_ecoc_validity(ecoc_matrix)
                score = np.mean(distance.cdist(ecoc_matrix, ecoc_matrix, metric='hamming'))
                if score > best_score:
                    best_score = score
                    best_ecoc_matrix = ecoc_matrix

            ecoc_matrix = best_ecoc_matrix




        else:
            raise ValueError('ecoc_type should be ovo ,ova, random_sparse or random_dense')
        return ecoc_matrix


    def _check_ecoc_validity(self, ecoc_matrix):
        #  check at least one -1 , one 1 and one 0 in each column
        for i in range(ecoc_matrix.shape[1]):
            column = ecoc_matrix[:, i].copy()
            if self.ecoc_type == 'random_sparse':
                if not (-1 in column):
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

                if not (0 in column):
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
                if not (1 in column):
                    indices = np.where(column == -1)[0]
                    random_index = np.random.choice(indices, size=1, replace=False)
                    column[random_index] = 1
                if not (-1 in column):
                    indices = np.where(column == 1)[0]
                    random_index = np.random.choice(indices, size=1, replace=False)
                    column[random_index] = -1

            ecoc_matrix[:, i] = column
        return ecoc_matrix

    def decode_labels(self, predictions):
        # print ("decoding predictions: ", predictions)
        # print(self.ecoc_matrix)
        min_distances = []
        for pred_row in predictions:
            if self.decoding_method == 'hamming':
                hamming_distances = distance.cdist([pred_row], self.ecoc_matrix, metric='hamming')
                min_distance_index = np.argmin(hamming_distances)
                min_distances.append(min_distance_index)
            elif self.decoding_method == 'euclidean':
                distances = distance.cdist([pred_row], self.ecoc_matrix, metric='euclidean')
                min_distance_index = np.argmin(distances)
                min_distances.append(min_distance_index)
            elif self.decoding_method == 'modified_hamming':
                output_code=pred_row.copy()
                reject_index=np.where(pred_row==0)[0]
                output_code[reject_index]=-1
                distances = distance.cdist([output_code], self.ecoc_matrix, metric='hamming')
                minus_one_index=np.argmin(distances)
                minus_one_class_code=self.ecoc_matrix[minus_one_index,:]

                output_code=pred_row.copy()
                output_code[reject_index]=1
                distances = distance.cdist([output_code], self.ecoc_matrix, metric='hamming')
                one_index=np.argmin(distances)
                one_class_code=self.ecoc_matrix[one_index,:]


                non_regjected_indices=np.where(pred_row!=0)[0]

                if  distance.hamming(pred_row[non_regjected_indices],minus_one_class_code[non_regjected_indices])<distance.hamming(pred_row[non_regjected_indices], one_class_code[non_regjected_indices]):
                    min_distances.append(minus_one_index)
                else:
                    min_distances.append(one_index)

            elif self.decoding_method == 'modified_hamming2':
                non_rejected_indices=np.where(pred_row!=0)[0]
                distances = distance.cdist([pred_row[non_rejected_indices]], self.ecoc_matrix[:,non_rejected_indices], metric='euclidean')
                min_distance_index = np.argmin(distances)
                min_distances.append(min_distance_index)





        return min_distances


if __name__ == '__main__':
    print("{:<20}            {:<20}          {:<20}     {:<20}    {:<10}".format("data", "ECOC accuracy",
                                                                                 "conformed ECOC accuracy",
                                                                                 "difference", "number of classes"))
    np.random.seed(42)
    directory = 'data/'
    # Pattern to match CSV files
    file_pattern = '*.csv'
    # Iterate over CSV files in the directory
    for data_path in glob.glob(os.path.join(directory, file_pattern)):
        print(data_path)
        # data_path="data/shuttle.csv"
        if data_path in ['data/soybean.csv','data/zoo.csv',"data/satimage.csv","data/shuttle.csv","data/ecoli.csv"]:
            continue
        df = pd.read_csv(data_path, header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

        n_classes = len(np.unique(y))

        data_dic = {"train": (X_train, y_train), "test": (X_test, y_test), "val": (X_val, y_val)}
        estimator= LogisticRegression(max_iter=3000,random_state=42)
        # estimator = SVC(kernel='linear', probability=True,random_state=42)
        ecoc = TernaryECOC(n_classes, data_dic, estimator,
                           with_conformal_prediction=True, ecoc_type="ovo",decoding_method="modified_hamming2")

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
