from get_data import Get_data
import albumentations as A
from Custom_Dataset import CustomDataset
import torch

class Loaders:
    def __init__(self, train_path:str, test_path:str, transform:A.Compose):
        '''
        params:
            batch_size: batch size
            train_path: path to training data
            test_path: path to test data
            transform: transform to be applied to the data
        '''
        self.train_path = train_path
        self.test_path = test_path
        self.transform = transform

        get_data = Get_data(train_path, test_path)
        self.X_train, self.X_test, self.y_train = get_data.get_data()
        
    def get_loader(self, type:str, batch_size:int)->torch.utils.data.DataLoader:
        '''
            params:
                type: 'train', 'test'
                batch_size: batch size
            return: Python DataLoader
        '''
        if type == 'train':
            train_dataset = CustomDataset(self.X_train, self.y_train, transform=self.transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            return train_loader
        elif type == 'test':
            test_dataset = CustomDataset(self.X_test, self.X_test, transform=None)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            return test_loader
        else:
            raise ValueError('type must be \'train\' or \'test\'')
        


if __name__ == '__main__':
    train_path = "../data/training.csv"
    test_path = "../data/test.csv"

    #transform
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(p=0.5, limit=10),
            A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.1, rotate_limit=10)
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    )

    #loaders
    loaders = Loaders(train_path, test_path, transform)
    train_loader = loaders.get_loader('train', batch_size=32)
    test_loader = loaders.get_loader('test', batch_size=32)
