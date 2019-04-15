import numpy as np
import tensorflow as tf
from Preprocessing import save_expanded, encode_oneHot, load_embeddings, convert_format, closest_vectors
#import tensorflow.keras.datasets as datasets  
from model import zsModel
import matplotlib.pyplot as plt
import sklearn.manifold as fold
from cycleGAN import zsCycle

def main():

    train = True 
    checkResults = False
    test = False
    
    #save_expanded()

    zsl_dict = None
    train_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    zsl_classes = ['bicycle', 'helicopter', 'submarine']
    seen_labels = []
    relevant_classes = train_classes + zsl_classes
    zsl_dict = load_embeddings("glove.6B.100d.txt", train_classes, zsl_classes)    
    
    x_train_a = np.load("data/trainA.npy")
    x_train_b = np.load("data/trainB.npy")
    y_train_a = np.load("data/expandedTrainY.npy")
            
    plt.imshow(x_train_a[0])
    plt.imshow(x_train_a[1])
    print(x_train_b.shape)
    print('--------------')
    
    hparams = tf.contrib.training.HParams(input_shape = (None, 160, 160, 3),
                                          embed_shape = (None, 100),
                                          batch_size = 1,
                                          name = "cycle_Adam_v3_3_MSE",
                                          lr = (1e-4)*2,
                                          beta1 = 0.5,
                                          beta2 = 0.999
                                          )
    #v2_1 = 1e-3 instance noise
    
    #v1 consists of several factors: 
    #Label smoothing 
    #EMA (Historical Averaging)
    #v1 Exclusive: Instance Norm (seems to break gradient penalty)
    
    #v2:
    #Instance noise
    #Gradient Penalty
    
    #v3:
    #Spectral Normalization
    #Batch size 8
    #New: Regular 5000 step sanity check based on l2 distance
    
    #exp:
    #Restricted discriminator training (trains 2000 steps (batches) before stopping until generator loss catches up)
    #Removed additional layers for transpose convolutions (embeddings -> image)
    #mse loss used in place of sce

    
    newModel = zsCycle(hparams)

    #closest_vectors("glove.6B.100d.txt", train_classes, zsl_dict)
    embed_matrix = np.load("embedding_matrix.npy")
    #print(embed_matrix.shape)
    #newModel.load_weights("cycle_Adam_v3_3_MSE.ckpt-13300")

    if train:
    
        newModel.train(x_train_a, y_train_a, x_train_b, 100, zsl_dict, train_classes)
        print("FINISHED TRAINING")    
        
        newModel.save(0)
    
    sampleImg = x_train_a[np.arange(5)]
    sampleImg_B = x_train_b[np.arange(5)]
    
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
    
    if checkResults:
    
        rand_idxs = np.random.randint(0, x_train_a.shape[0], 100)
        selected_imgs = x_train_a[rand_idxs]
        associated_labels = y_train_a[rand_idxs]
        embed_list = []
        l2_losses = []
        
        for l, v in zip(associated_labels, selected_imgs):
            l = l[0]
            plt.imshow(v)
            plt.show()
            v = np.expand_dims(v, 0)
            pred_emb = newModel.predict(v)
            embed_list.append(pred_emb)
            
            #L2 error
            l2_losses.append(np.linalg.norm(pred_emb - zsl_dict[train_classes[l]]))
            
            print("Associated Label: {}".format(train_classes[l]))
            
        plt.xlabel("Selected Embeddings")
        plt.ylabel("L2 difference")
        
        plt.plot(l2_losses)
        plt.show()    
    
    #Outdated for the most part
    if test:
    
        relevant_embeds = []
        
        for pred_class in relevant_classes:
            relevant_embeds.append(zsl_dict[pred_class]) #In order of relevant classes
        relevant_embeds = np.array(relevant_embeds)
        x_embed = fold.TSNE(n_components=2, init = 'pca', n_iter = 2000).fit_transform(relevant_embeds)
    
        reshaped_embed = np.zeros((2,len(relevant_classes)))
        
        for i,v in enumerate(x_embed):
            reshaped_embed[0][i] = v[0]
            reshaped_embed[1][i] = v[1]
        
        for i in range(reshaped_embed.shape[-1]):
            plt.scatter(reshaped_embed[0][i], reshaped_embed[1][i])
            plt.annotate(relevant_classes[i], (reshaped_embed[0][i], reshaped_embed[1][i]))
            
        plt.show()
    
    
    
    if test:
        totalImg = []
        totalLabel = []
        #newModel.accuracy_test(x_test, y_test)
        
        #Read in images
        for zclass in zsl_classes:
            path = "zsImages/"
            img, label = convert_format(path+zclass+"s", zclass)
            totalImg += img
            totalLabel += label
            
        zsl_accuracies = dict()
        zsl_accuracies_k = dict()
        for zclass in zsl_classes:
            zsl_accuracies[zclass] = 0
            zsl_accuracies_k[zclass] = 0

        once = True
        
        for img, label in zip(totalImg, totalLabel):
            
            euc_distances = []
            features = newModel.featurePredicton(np.expand_dims(img, axis = 0))
            features = np.squeeze(features)
            
            #Visualize vectors with tsne dim reduction

            if label not in seen_labels:
                relevant_embeds = []
                for pred_class in relevant_classes:
                    relevant_embeds.append(np.array(zsl_dict[pred_class]))
                relevant_embeds.append(list(features)) #Maintain format
                relevant_embeds = np.array(relevant_embeds)
                print(relevant_embeds.shape)
                
                x_embed = fold.TSNE(n_components=2, init = 'pca', n_iter = 2000).fit_transform(relevant_embeds)
                reshaped_embed = np.zeros((2,len(relevant_classes)+1))
                
                #Reshape
                for i,v in enumerate(x_embed):
                    reshaped_embed[0][i] = v[0]
                    reshaped_embed[1][i] = v[1]
                reshaped_embed = np.array(reshaped_embed)
                
                #Plot prediction
                for i in range(reshaped_embed.shape[-1]):
                    plt.scatter(reshaped_embed[0][i], reshaped_embed[1][i])
                    if (i < len(relevant_classes)):
                        plt.annotate(relevant_classes[i], (reshaped_embed[0][i], reshaped_embed[1][i]))
                plt.show()
                seen_labels.append(label)
                print(label)
              
            #Minimize euclidean distance to nearest word vector
            for test_class in zsl_dict.keys():
                if once:
                    print(newModel.test_layer(np.expand_dims(img, axis = 0)))
                    print(features)
                    print(np.array(zsl_dict[test_class]))
                    once = False
                euc_distances.append(np.linalg.norm(features-(np.array(zsl_dict[test_class]))))
                
            euc_array = np.array(euc_distances)
            min_idx = np.argmin(euc_array)
            k_smallest = np.argpartition(euc_array, 3)[:3]
             
            #Calculuate accuracies
            for i,v in enumerate(zsl_dict.keys()):
                if (i == min_idx):
                    if (v == label):
                        zsl_accuracies[label] += 1
                        
            for i, v in enumerate(zsl_dict.keys()):
                if i in k_smallest:
                    if v == label:
                        zsl_accuracies_k[label] += 1
                
        for key in zsl_accuracies.keys():
            zsl_accuracies[key] /= totalLabel.count(key)
            print("ZSL Accuracy for {} is: {}".format(key, zsl_accuracies[key]))
            zsl_accuracies_k[key] /= totalLabel.count(key)
            print("ZSL Accuracy (Top 3) for {} is {}".format(key, zsl_accuracies_k[key]))
    
if __name__ == "__main__":
    main()