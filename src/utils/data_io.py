"""
    Implements all io operations of data.
"""

from utils._libs_ import np, pd, torch, Variable
import json

# -------------------------------------------------------------------------------------------------------------------------------------------------
"""
Get the data generator
"""
def getGenerator(data_name):
    return GeneralGenerator

# -------------------------------------------------------------------------------------------------------------------------------------------------
"""
DataGenerator class produces data samples for all models
"""
class DataGenerator():
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, data_dict, train_share=(.8, .1), input_T=10,main_taskpoint=0,
                 task_span=1, limit=np.inf, cuda=False):

        self.data_dict=data_dict
        if limit < np.inf: self.X = self.X[:limit]
        self.train_share = train_share
        self.input_T = input_T
        self.main_taskpoint= main_taskpoint
        self.task_span = task_span
        self.row_num = np.array(self.data_dict[0]['X']).shape[0]
        self.column_num = np.array(self.data_dict[0]['X']).shape[1]
        self.cuda = cuda
        self.split_data()
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Split the training, validation and testing data
    """
    def split_data(self):
        train_dict,test_dict=[],[]
        for d in self.data_dict:
             if d['watch'] == 0:
                 test_dict.append(d)
             else:
                 train_dict.append(d)
        self.train_set = self.batchify(train_dict,test_dict)
        print('test:',self.train_set[0].shape)
        test_X=torch.from_numpy(np.array(test_dict[0]['X']))
        test_X=torch.unsqueeze(test_X,dim=0)
        test_Y=torch.from_numpy(np.array(test_dict[0]['y']))
        self.test_set = [test_X,test_Y]
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Get the X and Y for each set
    """
    def batchify(self, setDict,testdict):
        """
        Arguments:
            setDict  - 数据字典，要训练跟测试的数据
        Returns:
            [X, Y]
        """
        idx_num = np.array(setDict).shape[0]
        X = torch.zeros((idx_num, self.input_T, self.column_num))
        Y = torch.zeros((idx_num, self.main_taskpoint* 2+1, self.column_num))
        for i in range(idx_num ):
            X[i, :, :] = torch.from_numpy(np.array(setDict[i]['X']))
            Y[i, :, :] = torch.from_numpy(np.array(setDict[i]['y']))       
        return [X, Y]
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Get the batch data
    """
    def get_batches(self, X, Y, batch_size, shuffle=True):
        """
        Arguments:
            X            - (torch.tensor) input dataset
            Y            - (torch.tensor) ground-truth dataset
            batch_size   - (int)          batch size
            shuffle      - (boolean)      whether shuffle the dataset
        Yields:
            (batch_X, batch_Y)
        """
        length = len(X)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            batch_X = X[excerpt]
            batch_Y = Y[excerpt]
            if (self.cuda):
                batch_X = batch_X.cuda()
                batch_Y = batch_Y.cuda()
            yield Variable(batch_X), Variable(batch_Y)
            start_idx += batch_size

# -------------------------------------------------------------------------------------------------------------------------------------------------
"""
A GeneralGenerator reads complete data from .txt file without outliers 
"""
class GeneralGenerator(DataGenerator):
     def __init__(self, data_path, train_share=(.8, .1), input_T=10, main_taskpoint=0,
                  task_span=1, limit=np.inf, cuda=False):
        data_dict = []
        with open(data_path, 'r') as fj:
             for line in fj.readlines():
                 data_dict.append(json.loads(line))
        fj.close()
        super(GeneralGenerator, self).__init__(data_dict, 
                                               train_share=train_share,
                                               input_T=input_T,
                                               main_taskpoint=main_taskpoint,
                                               task_span=task_span,
                                               limit=limit,
                                               cuda=cuda)