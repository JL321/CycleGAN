import numpy as np
import tensorflow as tf
from Preprocessing import save_expanded, encode_oneHot, load_embeddings, convert_format, closest_vectors
#import tensorflow.keras.datasets as datasets  
from model import zsModel
import matplotlib.pyplot as plt
from cycleGAN import zsCycle

def main():

    train = True 
    checkResults = False
    test = False
    
    #save_expanded()

    zsl_dict = None
    train_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    seen_labels = []
    relevant_classes = train_classes 
    
    x_train_a = np.load("data/trainA.npy")
    x_train_b = np.load("data/trainB.npy")
    y_train_a = np.load("data/expandedTrainY.npy")
    
    hparams = tf.contrib.training.HParams(input_shape = (None, 160, 160, 3),
                                          embed_shape = (None, 100),
                                          batch_size = 1,
                                          name = "cycleGAN",
                                          lr = (1e-4)*2,
                                          beta1 = 0.5,
                                          beta2 = 0.999
                                          )
    
    newModel = zsCycle(hparams)
    if train:
    
        newModel.train(x_train_a, y_train_a, x_train_b, 100 train_classes)
        print("FINISHED TRAINING")    
        newModel.save(0)
    
    sampleImg = x_train_a[np.arange(5)]
    sampleImg_B = x_train_b[np.arange(5)]
    
    #Output sample images
    predImg = newModel.predict(sampleImg)
    predA = newModel.predict(sampleImg_B)
    for img in predImg:
        plt.imshow(img)
        print('------------------------')
        plt.show()
        
    for img_A in predA:
        plt.imshow(img_A)
        print('------------------------')
        plt.show()
if __name__ == "__main__":
    main()
