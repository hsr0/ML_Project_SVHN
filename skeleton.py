from train import _train
from inference import Infer
import numpy as np
from PIL import Image
from preprocess import PreProcess
import pickle
import gzip

class SVHN(object):

    path = ""

    def __init__(self, data_dir):
        """
            data_directory : path like /home/rajat/mlproj/dataset/
                            includes the dataset folder with '/'
            Initialize all your variables here
        """
        self.data_path = data_dir
        self.logs = 'logs/'
        self.checkpoint = './latest.ckpt'
        self.infer = Infer(self.checkpoint)
        self.init = False

    def train(self):
        """
            Trains the model on data given in path/train.csv

            No return expected
        """
        test = PreProcess(self.data_path)
        test.prepare_Data()
        _train(self.data_path, self.logs, self.checkpoint)

    def get_sequence(self, image):
        """
            image : a variable resolution RGB image in the form of a numpy array

            return: list of integers with the sequence of digits. Example: [5,0,3] for an image having 503 as the sequence.

        """
        if not self.init:
            self.infer.init_graph()
            self.init = True
        return self.infer.inferi(image)



    def save_model(self, **params):

        file_name = params['name']
        pickle.dump(self, gzip.open(file_name, 'wb'))

        """
            saves model on the disk

            no return expected
        """

    @staticmethod
    def load_model(**params):

        #file_name = params['name']
        #return pickle.load(gzip.open(file_name, 'rb'))
        return SVHN('/data')
        """
            returns a pre-trained instance of SVHN class
        """

if __name__ == "__main__":
        obj = SVHN('data/')
        #obj.save_model(name="svhn.gz")
        ttt = SVHN.load_model(name="svhn.gz")
        t = Infer('./latest.ckpt')
        for i in range(200):
            print(i)
            print(ttt.get_sequence(np.array(Image.open('25.png'))))
        #print(ttt.get_sequence(np.array(Image.open('1.png'))))
        for i in range(200):
            print(i)
            print(ttt.get_sequence(np.array(Image.open('7.png'))))
