
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from os.path import exists 


class FashionMNISTdataHandler:
    def __init__(self, path:str = "./FashionMNIST", transformer = None):
        '''
            initiates the Data handler
            @ params: 
                path : A path to the saved dataset or the desired path to save the dataset
                transformer: A preprocessor for the dataset before feeding to the DataLoader
        '''
        self.path = path
        download = True
        self.trainLoader= None
        self.testLoader = None
        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer = transforms.ToTensor ()
            
        if exists(self.path):
            download = False
        
        
        self.train_set = datasets.FashionMNIST(root =self.path, train = True ,
                                                            download = download , transform = self.transformer)
        self.test_set = datasets.FashionMNIST(root =self.path, train = False ,
                                                            download = download , transform = self.transformer)
    
    def __call__(self, batchSize:int = 32, shuffle: bool = True, NumWorkers: int= 1 ):
        '''
            @params:
                batchsize: the number of inputs for each batch to contain
                shuffle: whether to shuffle or not
                NumWorkers: How many Threads to split the process into
        '''
        self.trainLoader = DataLoader(self.train_set , batch_size = batchSize, shuffle = shuffle)#, num_workers = NumWorkers)
        self.testLoader = DataLoader(self.test_set , batch_size = batchSize, shuffle = shuffle)#, num_workers = NumWorkers) 
        return self.trainLoader,self.testLoader
