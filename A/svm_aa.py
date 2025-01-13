import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

import medmnist
from medmnist import INFO, BreastMNIST

import os

class modelASVM():
    def __init__(self, load = False):
        self.clf = None

        self.images, self.labels, self.dataset_sizes, self.class_names = self.load_dataset()

        if not load:
            self.clf, self.best_params = self.create_classifier()
        else:
            self.clf = self.load_model()

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
                #   "degree": [3, 4, 5, 6],
                  "gamma": [0.01, 0.1, "auto", "scale"],
                  "kernel": ["linear", "poly", "rbf", "sigmoid"],
                  "class_weight": [None, "balanced", {0: 0.7, 1: 0.3}]
                  }
        
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

        self.plot_accuracy(scores)
        return best_svm, best_params
    
    def plot_accuracy(self, scores):
        plt.figure()
        plt.plot(range(1, len(scores) + 1), scores, marker='o', label="Scores")
        plt.title("Highest accuracy for each iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        # plt.ylim(0,1)
        plt.grid()
        # plt.show()
    
    def get_test_accuracy(self):
        acc = self.clf.score(self.images["test"], self.labels["test"])
        predictions = self.clf.predict(self.images["test"])
        cm = confusion_matrix(self.labels["test"], predictions)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap=plt.cm.Blues)

        return acc
    
    def save_model(self, filename = "svm_model_a.pkl"):
        # if file extension is wrong or no file extension
        if not filename.endswith(".pkl"):
            print(f"Incorrect or missing file extension. Changing filename to '{filename}.pkl'")
            # removes any other file extension and adds .pkl
            filename = f"{os.path.splitext(filename)[0]}.pkl" 
        
        filepath = os.path.join(os.getcwd, "A", filename)
        if self.clf is None:
            raise ValueError("There is no model saved")

        joblib.dump(self.clf, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self):
        filename = str(input("Enter the filename \n"))

        # if file extension is wrong or no file extension
        if not filename.endswith(".pkl"):
            print(f"Incorrect or missing file extension. Changing filename to '{filename}.pkl'")
            # removes any other file extension and adds .pkl
            filename = f"{os.path.splitext(filename)[0]}.pkl" 

        filepath = os.path.join(os.getcwd(), "A", filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
    
        model = joblib.load(filepath)
        print(f"Model is loaded from {filepath}")
        return model

if __name__ == "__main__":
    new = modelASVM(load=False)
    test_acc = new.get_test_accuracy()

    print(test_acc)
    # print(new.clf.get_params())
   
    print(new.best_params)
    # new.save_model()

    plt.show()
