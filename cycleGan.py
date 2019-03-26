import tensorflow as tf
import numpy as np
from util import resnet_layer, batch_layer, instance_conv
import tensorflow.contrib.layers as layer
import random
from knownModels import build_generator_resnet_9blocks, build_gen_discriminator
import matplotlib.pyplot as plt

base_size = (None, 150, 150, 3)
embed_size = (None, 100)

def standard_gen(x, name):
    
    with tf.variable_scope(name):

        z = instance_conv(x, 64, 7)
        z = instance_conv(z, 128, 3, stride = 2)
        z = instance_conv(z, 256, 3, stride = 2)
        for _ in range(9):
            z = resnet_layer(z, [[256,3],[256,3]])
        
        z = tf.contrib.layers.conv2d_transpose(z, 128, 3, stride = 2, padding = 'SAME')
        z = tf.contrib.layers.conv2d_transpose(z, 64, 3, stride = 2, padding = 'SAME')
        z = tf.contrib.layers.conv2d_transpose(z, 3, 7, padding = 'SAME', activation_fn = None)
        out = tf.nn.tanh(z)
        
        return out
        
def standard_disc(x, name):
    
    with tf.variable_scope(name):
        
        z = instance_conv(x, 64, 4, stride = 2)
        z = instance_conv(z, 128, 4, stride = 2)
        z = instance_conv(z, 256, 4, stride = 2)
        z = instance_conv(z, 512, 4, stride = 2)
        z = tf.contrib.layers.conv2d(z, 1, 4, activation_fn = tf.nn.sigmoid)
        
        return z
    
def sce_loss(logits, labels):
    #Convenience function for sigmoid cross entropy
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    
def mse_loss(logits, labels):
    return tf.reduce_mean(tf.squared_difference(logits,labels))

class zsCycle:

    def __init__(self, hparams):
        
        self.input_shape = hparams.input_shape
                
        self.embed_shape = hparams.embed_shape
        self.name = hparams.name
        self.batch_size = hparams.batch_size
        self.lr = hparams.lr
        self.beta1 = hparams.beta1
        
        self.generator = standard_gen
        self.reverse_generator = standard_gen
        self.discriminator = standard_disc
        self.reverse_discriminator = standard_disc
        
        self.fake_image_A = np.zeros((50, 150, 150, 3))
        self.fake_image_B = np.zeros((50, 150, 150, 3))

        self.num_fake = 0
        
        self.build_graph()
        
    def build_graph(self):
        
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            self.sess = tf.Session(config=config)
            
            #Placeholders for fake and real inputs
            #A = image, B = embedding
            #Generator: img -> embed
            #Reverse_generator: embed -> img
            #Discriminator: img
            #Reverse_D: embed
                            
            self.img_A = tf.placeholder(tf.float32, (self.input_shape))
            self.embed_B = tf.placeholder(tf.float32, (self.embed_shape))
         
            self.fake_A_sample = tf.placeholder(tf.float32, (self.input_shape))
            self.fake_B_sample = tf.placeholder(tf.float32, (self.embed_shape))
            
            with tf.variable_scope(self.name) as scope:
                                
                #Generator Discriminator Ops
                
                self.fake_B = self.generator(self.img_A, 'g_A')
                self.fake_A = self.reverse_generator(self.embed_B, 'g_B')

                self.DA_real = self.discriminator(self.img_A, 'd_A')
                self.DB_real = self.reverse_discriminator(self.embed_B, 'd_B')

                scope.reuse_variables()

                self.fake_A2B = self.generator(self.fake_A, 'g_A')
                self.fake_B2A = self.reverse_generator(self.fake_B, 'g_B')

                self.DA_fake = self.discriminator(self.fake_A, 'd_A')
                self.DB_fake = self.reverse_discriminator(self.fake_B, 'd_B')
                
                scope.reuse_variables()

                #s representing sample
                self.DA_fake_s = self.discriminator(self.fake_A_sample, 'd_A')
                self.DB_fake_s = self.reverse_discriminator(self.fake_B_sample, 'd_B')
                
            self.loss_init()
            
    def loss_init(self):
        
        self.cycle_loss = tf.reduce_mean(tf.abs(self.fake_A2B-self.embed_B))+tf.reduce_mean(tf.abs(self.fake_B2A-self.img_A))

        #Generator losses
        self.disc_loss_A = sce_loss(self.DA_fake,tf.ones_like(self.DA_fake))
        self.disc_loss_B = sce_loss(self.DB_fake,tf.ones_like(self.DB_fake))
        
        self.gen_loss_b = self.disc_loss_A + 10*self.cycle_loss
        self.gen_loss_a = self.disc_loss_B + 10*self.cycle_loss
        
        #Discriminator losses
        
        self.disc_loss_real_a = sce_loss(self.DA_real, tf.ones_like(self.DA_real))
        self.disc_loss_fake_a = sce_loss(tf.square(self.DA_fake_s), tf.zeros_like(self.DA_fake_s))
        self.disc_loss_a = (self.disc_loss_real_a + self.disc_loss_fake_a)/2
        
        self.disc_loss_real_b = sce_loss(self.DB_real, tf.ones_like(self.DB_real))
        self.disc_loss_fake_b = sce_loss(tf.square(self.DB_fake_s), tf.zeros_like(self.DB_fake_s))
        self.disc_loss_b = (self.disc_loss_real_b + self.disc_loss_fake_b)/2
                                       
        #Tensorboard
        
        self.g_loss_as = tf.summary.scalar("g_loss_a", self.gen_loss_a)
        self.g_loss_bs = tf.summary.scalar("g_loss_b", self.gen_loss_b)
        self.cycle_loss_s = tf.summary.scalar('g_loss_cycle', self.cycle_loss)
        
        #self.g_sum = tf.summary.merge([self.g_loss_as, self.g_loss_bs, self.cycle_loss_s])
        
        self.d_loss_sum_as = tf.summary.scalar('d_loss_a', self.disc_loss_a)
        self.d_loss_sum_bs = tf.summary.scalar('d_loss_b', self.disc_loss_b)
        #self.d_sum = tf.summary.merge([self.d_loss_sum_as, self.d_loss_sum_bs])
        
        self.g_A_var = [v for v in tf.trainable_variables() if 'g_A' in v.name]
        self.g_B_var = [v for v in tf.trainable_variables() if 'g_B' in v.name]
        self.d_A_var = [v for v in tf.trainable_variables() if 'd_A' in v.name]
        self.d_B_var = [v for v in tf.trainable_variables() if 'd_B' in v.name]
        
        print(len(self.g_A_var))
        #for v in tf.trainable_variables():
            #print(v.name)
               
        optimizer = tf.train.GradientDescentOptimizer(self.lr)

        self.train_gen_a = optimizer.minimize(self.gen_loss_a, var_list = self.g_A_var)
        self.train_disc_a = optimizer.minimize(self.disc_loss_a, var_list = self.d_A_var)
        self.train_gen_b = optimizer.minimize(self.gen_loss_b, var_list = self.g_B_var)
        self.train_disc_b = optimizer.minimize(self.disc_loss_b, var_list = self.d_B_var)
            
        self.saver = tf.train.Saver()

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        ''' Temporary storage of fake images'''

        if(num_fakes < 50):
            fake_pool[num_fakes] = fake
            return fake
        else:
            rand_idx = np.random.randint(0, 50)
            p = random.random()
            if p > 0.5:
                temp = fake_pool[rand_idx]
                fake_pool[rand_idx] = fake 
                return temp
            else:
                return fake
    
    def train(self, data_A, data_B, epochs, continue_train = False):
        
        #data_A -> image type A
        #data_B -> image type B
        
        print("INITIALIZING TRAINING WITH MODEL {}".format(self.name))
        #Data_embed- collection of arbitrary word vectors
        max_len = data_A.shape[0]
        step = 0
        with self.graph.as_default():
        
            if (continue_train):
                self.saver.restore(self.sess, "model/{}.ckpt-6700".format(self.name))
                        
            self.writer = tf.summary.FileWriter("./logs/ZSLCycleSGD", self.sess.graph)

            self.sess.run(tf.global_variables_initializer())
            
            
            
            done_epoch = False

            for epoch in range(epochs):
                start_idx = 0
                end_idx = self.batch_size
                while done_epoch == False:
                                        
                    batch_imgA = []
                    batch_embedB = []

                    if (end_idx+1 > max_len):
                        print("END")
                        end_idx = max_len
                        done_epoch = True
                    
                    for i in range(start_idx, end_idx):
                        rand_idx = np.random.randint(0, data_B.shape[0])
                        batch_imgA.append(data_A[i])
                        batch_embedB.append(data_B[rand_idx])
                    
                    start_idx += self.batch_size
                    end_idx += self.batch_size
                    
                    batch_imgA = np.array(batch_imgA)
                    batch_embedB = np.array(batch_embedB)
                    
                    if (len(batch_imgA.shape) == 3):
                        batch_imgA = np.expand_dims(batch_imgA, axis = 0)
                    
                    if (len(batch_embedB.shape) == 1):
                        batch_embedB = np.expand_dims(batch_embedB, axis = 0)
                    
                    #Training ops and writing to tensorboard
                    
                    #Tensorboard ops
                    
                    _, fakeB, sum_loss = self.sess.run([self.train_gen_a, self.fake_B, self.g_loss_as],
                                                       feed_dict = {self.img_A: batch_imgA, self.embed_B: batch_embedB})
                    
                    fakeB_sample = self.fake_image_pool(self.num_fake, fakeB, self.fake_image_B)
                                             
                    if (len(fakeB_sample.shape) == 1):
                        fakeB_sample = np.expand_dims(fakeB_sample, axis = 0)
                    
                    self.writer.add_summary(sum_loss, step)
                    
                    _, sum_loss = self.sess.run([self.train_disc_b, self.d_loss_sum_bs],
                                                feed_dict = {self.img_A: batch_imgA, self.embed_B: batch_embedB, self.fake_B_sample: fakeB_sample})
                    
                    
                    self.writer.add_summary(sum_loss, step)
                    
                    _, fakeA, sum_loss, cycle_loss, b_loss = self.sess.run([self.train_gen_b, self.fake_A, self.g_loss_bs, self.cycle_loss_s, self.disc_loss_B],
                                                       feed_dict = {self.img_A: batch_imgA, self.embed_B: batch_embedB})
                    
                    fakeA_sample = self.fake_image_pool(self.num_fake, fakeA, self.fake_image_A)
                    
                    if (len(fakeA_sample.shape) == 3):
                        fakeA_sample = np.expand_dims(fakeA_sample, axis = 0)
                    
                    self.writer.add_summary(sum_loss, step)
                    self.writer.add_summary(cycle_loss, step)
                    
                    _, sum_loss = self.sess.run([self.train_disc_a, self.d_loss_sum_as],
                                                feed_dict = {self.img_A: batch_imgA, self.embed_B: batch_embedB, self.fake_A_sample: fakeA_sample})
    
                    self.writer.add_summary(sum_loss, step)
                    step += 1 
                    self.num_fake += 1

                                                
                np.random.shuffle(data_A)
                np.random.shuffle(data_B)
                done_epoch = False
                print("Saving model at {} in {}th epoch".format("model/"+self.name+".ckpt", epoch))
                self.save(step)

                print("Finished training {}th epoch".format(epoch))
            
    def save(self, step):
        
        model_path = "model/{}.ckpt".format(self.name)
        self.saver.save(self.sess, model_path, global_step = step)
            
    def load_weights(self, modelName):
        
        self.saver.restore(self.sess, "model/{}".format(modelName))
                    
