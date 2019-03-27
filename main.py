import numpy as np
import tensorflow as tf
from Preprocessing import save_expanded, encode_oneHot, load_embeddings, convert_format, closest_vectors
#import tensorflow.keras.datasets as datasets  
from model import zsModel
import matplotlib.pyplot as plt
import sklearn.manifold as fold
from cycleGan import zsCycle

def main():

    train = True    
    
    x_train_a = np.load("data/trainA.npy")
    x_train_b = np.load("data/trainB.npy")
            
    hparams = tf.contrib.training.HParams(input_shape = (None, 150, 150, 3),
                                          embed_shape = (None, 100),
                                          batch_size = 1,
                                          name = "ZSModelCycleAdam01",
                                          lr = 0.0002,
                                          beta1 = 0.5
                                          )
    
    newModel = zsCycle(hparams)
    #newModel.load_weights("ZSModelCycleGD.ckpt")
    #print("GRAPH BUILT")
    
    if train:
    
        newModel.train(x_train_a, x_train_b, 100)
        print("FINISHED TRAINING")    
        
        newModel.save(0)

if __name__ == "__main__":
    main()
