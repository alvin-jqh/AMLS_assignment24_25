import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import PCA

import medmnist
from medmnist import INFO, BreastMNIST

import os

class modelASVM():
    def __init__(self):
        self.clf = None

        self.images, self.labels, self.dataset_sizes, self.class_names = self.load_dataset()

        self.clf, self.best_params, self.scores = self.create_classifier()

    def process_images(self, images: np.ndarray):
        """
        Returns flattened images scaled between 0 and 1
        """
        data_shape = images.shape
        images = images.reshape(data_shape[0], data_shape[1] * data_shape[2])/255

        return images
    
    def reduce_data(self, data):
        pca = PCA(n_components=28)
        reduced_data = pca.fit_transform(data)

        return reduced_data

    def load_dataset(self):
        """
        Function to load the data the BreastMNIST data set into Numpy Array
        Returns:
            images: dictionary containing images for all 3 data splits, images are flattened
            labels: dictionary containing labels for all 3 data splits
            dataset_sizes: dictionary of split sizes
            class_names: class map dictionary
        """

        # get the path for the datasets
        dir = os.getcwd()

        data_flag = "breastmnist"
        data_dir = "Datasets"

        PATH = os.path.join(dir, data_dir)

        # retrieve the data from the 
        datasets = {x: BreastMNIST(split= x, transform=None, download=False, root=PATH)
                    for x in ["train", "val", "test"]} 
        
        # get the size of all 3 data splits
        dataset_sizes = {x: len(datasets[x]) for x in ["train", "val", "test"]}

        # process the images
        images = {x: self.process_images(datasets[x].imgs) for x in ["train", "val", "test"]}
        labels = {x: datasets[x].labels.flatten() for x in ["train", "val", "test"]}

        # load the class names
        info = INFO[data_flag]
        class_names = info["label"]
        class_names = {int(key): value for key, value in class_names.items()}

        return images, labels, dataset_sizes, class_names
    
    def create_classifier(self):
        """
        Performs cross validation to pick the best hyperparameters and kernels for the dataset
        """

        base_clf = svm.SVC()

        p_grid = {"C": [0.1, 1, 10, 100],
                  "degree": [3, 4, 5, 6],
                  "gamma": [0.01, 0.1, "auto", "scale"],
                  "kernel": ["linear", "poly", "rbf", "sigmoid"]}
        
        # track the highest scores for each iteration
        scores = []

        #  track best overall parameters and the model
        highest_score = 0
        best_svm = None
        best_params = None

        for i in range (5):
            inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)

            # cross validation 
            clf = GridSearchCV(estimator=base_clf, param_grid=p_grid, cv=inner_cv)

            # fit the svm
            clf.fit(np.concatenate((self.images["train"], self.images["val"]), axis=0), 
                    np.concatenate((self.labels["train"], self.labels["val"]), axis=0))
            
            scores.append(clf.best_score_)

            print(f"Iteration {i}, best score for this iteration is {clf.best_score_}")

            # update the values if the 
            if clf.best_score_ > highest_score:
                highest_score = clf.best_score_
                best_params = clf.best_params_
                best_svm = clf.best_estimator_

        return best_svm, best_params, scores
    
    def get_test_accuracy(self):
        acc = self.clf.score(self.images["test"], self.labels["test"])
        return acc
    

if __name__ == "__main__":
    new = modelASVM()
    print(new.get_test_accuracy())
    print(new.best_params)
