import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

class Get_data:
    def __init__(self, train_path:str, test_path:str):
        '''
        params:
            train_path: path to training data
            test_path: path to test data

        '''
        self.train_path = train_path
        self.test_path = test_path

    def get_image(self, Image:pd.Series)->np.array:
        '''
            params:
                Image: pandas series of image
            returns:
                image: numpy array of image
        '''
        images = []
        for i in range(len(Image)):
            img = Image[i].split(' ')
            img = np.array(img, dtype=float).reshape(96,96)
            images.append(img)
        return np.array(images)

    def get_data(self)->tuple:
        '''
        params:
            None
        returns:
            A tuple of training data, test data, and training labels
        '''
        df_train = pd.read_csv(self.train_path)
        df_test = pd.read_csv(self.test_path)
        lables_train = df_train.drop('Image', axis=1,)
        #fillna
        lables_train.fillna(lables_train.mean(), inplace=True)

        images_train = self.get_image(df_train['Image'])
        images_test = self.get_image(df_test['Image'])
        labels_train = lables_train.values

        images_train = torch.Tensor(images_train).view(-1, 1, 96, 96)
        images_test = torch.Tensor(images_test).view(-1, 1, 96, 96).float()
        labels_train = torch.Tensor(labels_train)
        #reshape labels_train to be (n, 15, 2)
        labels_train = labels_train.view(-1, 15, 2)
        #print(images_train.shape, images_test.shape, labels_train.shape)
        
        return images_train, images_test, labels_train      

    def plot_data(self, image:np.array, keypoints:np.array):
        '''
            params:
                image: numpy array of image
                label: numpy array of label
            returns:    
        '''
        keypoints = keypoints.reshape(30, 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.scatter(keypoints[::2], keypoints[1::2], s=10, marker='*', c='r')
        plt.show()
        

if __name__ == "__main__":
    train_path = "../data/training.csv"
    test_path = "../data/test.csv"

    #Get data
    get_data = Get_data(train_path, test_path)
    X_train, X_test, y_train = get_data.get_data()
    print(list(tuple(y_train[0].numpy())))

    #plot data
    get_data.plot_data(X_train[0], y_train[0])

    