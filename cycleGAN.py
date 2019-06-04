import tensorflow as tf
import numpy as np
from ops import resnet_layer, batch_layer, instance_conv, custom_dense, custom_conv2d, resnet_transpose
import tensorflow.contrib.layers as layer
import random
import matplotlib.pyplot as plt
from Preprocessing import cifar10_wlabel

base_size = (None, 160, 160, 3)
embed_size = (None, 100)

def batch_shape(main_shape, batch_size):
    converted_shape = list(main_shape[1:])
    converted_shape.insert(0, batch_size)
    converted_shape = tuple(converted_shape)
    return converted_shape

def discriminator(x, name):
    
    #discriminator for embeddings
    
    with tf.variable_scope(name):
       
        z = custom_dense(x, 256, True, tag = 'd1')
        z = custom_dense(z, 512, True, tag = 'd2')
        z = custom_dense(z, 256, True, tag = 'd_finalB_{}'.format(name))
        z = custom_dense(z, 1, False, activation_fn = tf.nn.sigmoid, tag = 'd4')
    
        return z
    
def reverse_discriminator(x, name):
    
    with tf.variable_scope(name):
        
        #discriminator for img
        #Resulting shape: input_shape/16
        z = custom_conv2d(x, 64, 4, stride = 2, tag = 'k1')
        z = custom_conv2d(z, 128, 4, stride = 2, tag = 'k2')
        z = custom_conv2d(z, 256, 4, stride = 2, tag = 'd_finalA_{}'.format(name))
        z = custom_conv2d(z, 512, 4, stride = 2, tag = 'k3')
        z = custom_conv2d(z, 1, 4, stride = 1, activation_fn = tf.nn.sigmoid, tag = 'k4')
        
        return z
        
def reverse_generator(x, name):
    
    #Converts an embedding vector into an image
    
    with tf.variable_scope(name):
        
        z = batch_layer(x, 1024)
        z = batch_layer(z, 512)
        z = layer.fully_connected(z, 256)
        z = tf.expand_dims(z, 1)
        z = tf.expand_dims(z, 1)
        
        #Size = 2   
        z = layer.conv2d_transpose(z, 256, 3, padding = 'VALID')
     
        #Size = 7
        z = layer.conv2d_transpose(z, 128, 3, stride = 2, padding = 'VALID')

        #Size = 15
        z = layer.conv2d_transpose(z, 128, 3, stride = 2, padding = 'VALID')
        
        #Size = 30
        z = layer.conv2d_transpose(z, 128, 3, stride = 2)        

        for i in range(3):
            if i == 2:
                z = resnet_transpose(z, scope = 'g_finalA_{}'.format(name))
            else:
                z = resnet_transpose(z)

        #Size = 60
        z = layer.conv2d_transpose(z, 3, 3, stride = 2, activation_fn = tf.nn.tanh)
        
        print("{} init, {} base".format(z.shape[1:], base_size))
        assert(z.shape[1:] == base_size[1:])
        #Produces 60x60x3 image
        
        return z
    
def generator(x, name):

    #Converts an image into an embedding vector
    
    with tf.variable_scope(name):
     
        z = layer.instance_norm(layer.conv2d(x, 64, 7))
        z = layer.instance_norm(layer.conv2d(z, 128, 3, stride = 2))
        z = layer.conv2d(z, 256, 3, stride = 2)
        for _ in range (3):
            z = resnet_layer(z, [[256, 3], [256, 3]])
        z = layer.flatten(z)
        z = layer.fully_connected(z, 1024)
        z = layer.fully_connected(z, 512)
        z = layer.fully_connected(z, 256, scope = 'g_finalB_{}'.format(name))
        z = layer.fully_connected(z, int(embed_size[1]), activation_fn = tf.nn.tanh)
        
        return z
    
def img2img_gen(x, name):
    
    with tf.variable_scope(name):
        
        z = layer.instance_norm(layer.conv2d(x, 64, 7))
        z = layer.instance_norm(layer.conv2d(z, 128, 3, stride = 2))
        z = layer.conv2d(z, 256, 3, stride = 2)
        for _ in range (6):
            z = resnet_layer(z, [[256, 3], [256, 3]])
        z = layer.instance_norm(layer.conv2d_transpose(z, 128, 3, stride = 2))
        z = layer.instance_norm(layer.conv2d_transpose(z, 64, 3, stride = 2))
        z = layer.instance_norm(layer.conv2d_transpose(z, 3, 7), activation_fn = tf.nn.tanh, scope = 'g_final_{}'.format(name))
        return z
            
    
def sce_loss(logits, labels):
    #Convenience function for sigmoid cross entropy
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    
def mse_loss(logits, labels):
    return tf.reduce_mean(tf.squared_difference(logits,labels))

class zsCycle:

    def __init__(self, hparams):
        
        self.eval_loss = mse_loss
        self.input_shape = hparams.input_shape
        self.embed_shape = hparams.embed_shape
        self.name = hparams.name
        self.batch_size = hparams.batch_size
        self.lr = hparams.lr
        self.beta1 = hparams.beta1
        self.beta2 = hparams.beta2
        
        self.generator = img2img_gen
        self.reverse_generator = img2img_gen
        self.reverse_discriminator = reverse_discriminator
        self.discriminator = reverse_discriminator
        
        self.fake_image_A = np.zeros((64, 160, 160, 3))
        self.fake_image_B = np.zeros((64, 160, 160, 3))
        self.fake_embed_B = np.zeros((64, 100))

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
            #reverse_discriminator: img
            #Reverse_D: embed
                            
            self.img_A = tf.placeholder(tf.float32, (self.input_shape))
            self.img_B = tf.placeholder(tf.float32, (self.input_shape))
            self.embed_B = tf.placeholder(tf.float32, (self.embed_shape))
         
            self.fake_A_sample = tf.placeholder(tf.float32, (self.input_shape))
            self.fake_B_sample = tf.placeholder(tf.float32, (self.input_shape))
            
            with tf.variable_scope(self.name) as scope:
                                
                #Generator reverse_discriminator Ops
                
                self.fake_A = self.generator(self.img_B, 'g_A')
                self.fake_B = self.reverse_generator(self.img_A, 'g_B')

                print(self.fake_B.shape)

                self.DA_real = self.reverse_discriminator(self.img_A, 'd_A')
                self.DB_real = self.discriminator(self.img_B, 'd_B')
                
                self.gGrad_Bb = tf.reduce_mean(tf.gradients(self.fake_B, self.img_A)) #Backpropagated variables
                self.gGrad_Ab = tf.reduce_mean(tf.gradients(self.fake_A, self.img_B))
                
                forward_varA = [v for v in tf.trainable_variables() if 'g_final_g_A' in v.name][0]
                forward_varB = [v for v in tf.trainable_variables() if 'g_final_g_B' in v.name][0]
                
                last_layer = tf.reduce_mean(forward_varA)
                self.last_layer = tf.summary.scalar("Last_Layer_A_Mean",last_layer)
                
                print("-A-")
                print(forward_varA.shape)
                
                print("-B-")
                print(forward_varB.shape)
                
                self.gGrad_Bf = tf.reduce_mean(tf.gradients(self.fake_B, forward_varB)) #Gradients on the final layers (immediate gradients)
                self.gGrad_Af = tf.reduce_mean(tf.gradients(self.fake_A, forward_varA))

                self.dGrad_Bb = tf.reduce_mean(tf.gradients(self.DB_real, self.img_B)) #Backpropagated variables
                self.dGrad_Ab = tf.reduce_mean(tf.gradients(self.DA_real, self.img_A))
                
                forward_varAD = [v for v in tf.trainable_variables() if 'd_finalA_d_A' in v.name][0]
                forward_varBD = [v for v in tf.trainable_variables() if 'd_finalA_d_B' in v.name][0]
                
                self.dGrad_Bf = tf.reduce_mean(tf.gradients(self.DB_real, forward_varBD)) #Gradients on the final layers (immediate gradients)
                self.dGrad_Af = tf.reduce_mean(tf.gradients(self.DA_real, forward_varAD))

                self.var_Bb = tf.summary.scalar('Back Gradient B Generator Mean', self.gGrad_Bb)       
                self.var_Bf = tf.summary.scalar('Forward Gradient B Generator Mean', self.gGrad_Bf)
                self.var_B = tf.summary.merge([self.var_Bb,self.var_Bf])
                
                self.var_Ab = tf.summary.scalar('Back Gradient A Generator Mean', self.gGrad_Ab)
                self.var_Af = tf.summary.scalar('Forward Gradient A Generator Mean', self.gGrad_Af)
                self.var_A = tf.summary.merge([self.var_Ab, self.var_Af])

                self.var_Bbd = tf.summary.scalar('Back Gradient B Discriminator Mean', self.dGrad_Bb)       
                self.var_Bfd = tf.summary.scalar('Forward Gradient B Discriminator Mean', self.dGrad_Bf)
                self.var_Bd = tf.summary.merge([self.var_Bbd,self.var_Bfd])
                
                self.var_Abd = tf.summary.scalar('Back Gradient A Discriminator Mean', self.dGrad_Ab)
                self.var_Afd = tf.summary.scalar('Forward Gradient A Discriminator Mean', self.dGrad_Af)
                self.var_Ad = tf.summary.merge([self.var_Abd, self.var_Afd])

                self.realRDisc = tf.gradients(tf.reduce_mean(self.DA_real), [self.img_A])
                realDisc = tf.gradients(tf.reduce_mean(self.DB_real), [self.img_B])

                #Gradient penalty coefficient currently set as 10

                self.gradient_penRDisc = 10*tf.square(tf.norm(self.realRDisc[0]))
                self.gradient_penDisc = 10*tf.square(tf.norm(realDisc[0]))

                scope.reuse_variables()

                self.fake_A2B = self.reverse_generator(self.fake_A, 'g_B')
                self.fake_B2A = self.generator(self.fake_B, 'g_A')

                '''
                self.fakeRDisc = tf.gradients(self.fake_A2B, self.fake_A)
                self.fakeDisc = tf.gradients(self.fake_B2A, self.fake_A)
                ^Gradient penalties from generator distribution
                '''
                
                self.DA_fake = self.reverse_discriminator(self.fake_A, 'd_A')
                self.DB_fake = self.discriminator(self.fake_B, 'd_B')
                
                scope.reuse_variables()

                #s representing sample
                self.DA_fake_s = self.reverse_discriminator(self.fake_A_sample, 'd_A')
                self.DB_fake_s = self.discriminator(self.fake_B_sample, 'd_B')
                
            self.loss_init()
            
    def loss_init(self):
        
        self.cycle_loss = tf.reduce_mean(tf.abs(self.fake_A2B-self.img_B))+tf.reduce_mean(tf.abs(self.fake_B2A-self.img_A))

        #Varied noisy labels that are separate for the generator and discriminator just in case
        add_onD = np.random.random()*0.1
        add_onG = np.random.random()*0.1
        noisy_labelG = add_onG + 0.9
        noisy_labelD = add_onD + 0.9
        
        A_label_sm = tf.constant(noisy_labelG, dtype = tf.float32, shape = batch_shape(self.DA_fake.shape, self.batch_size))
        B_label_sm = tf.constant(noisy_labelD, dtype = tf.float32, shape = batch_shape(self.DB_fake.shape, self.batch_size))

        #Generator losses
        self.gdisc_loss_A = self.eval_loss(self.DA_fake,A_label_sm)
        self.gdisc_loss_B = self.eval_loss(self.DB_fake,B_label_sm)

        self.gen_loss_b = self.gdisc_loss_B + 10*self.cycle_loss
        #a2b
        self.gen_loss_a = self.gdisc_loss_A + 10*self.cycle_loss
        #b2a
        
        #Discriminator losses
        
        disc_lA = tf.constant(noisy_labelD, dtype = tf.float32, shape = batch_shape(self.DA_real.shape, self.batch_size))
        self.disc_loss_real_a = self.eval_loss(self.DA_real, disc_lA)
        self.disc_loss_fake_a = self.eval_loss(self.DA_fake_s, tf.zeros_like(self.DA_fake_s))
        self.disc_loss_a = (self.disc_loss_real_a + self.disc_loss_fake_a)/2 + self.gradient_penRDisc
        
        disc_lB = tf.constant(noisy_labelD, dtype = tf.float32, shape = batch_shape(self.DB_real.shape, self.batch_size))
        self.disc_loss_real_b = self.eval_loss(self.DB_real, disc_lB)
        self.disc_loss_fake_b = self.eval_loss(self.DB_fake_s, tf.zeros_like(self.DB_fake_s))
        self.disc_loss_b = (self.disc_loss_real_b + self.disc_loss_fake_b)/2 + self.gradient_penDisc
                                       
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
               
        if ('Adam' in self.name):
            optimizer = tf.train.AdamOptimizer(self.lr, self.beta1)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.lr)

        train_gen_a = optimizer.minimize(self.gen_loss_a, var_list = self.g_A_var)
        train_disc_a = optimizer.minimize(self.disc_loss_a, var_list = self.d_A_var)
        train_gen_b = optimizer.minimize(self.gen_loss_b, var_list = self.g_B_var)
        train_disc_b = optimizer.minimize(self.disc_loss_b, var_list = self.d_B_var)
            
        ema = tf.train.ExponentialMovingAverage(decay = .99) #Up to tuning

        with tf.control_dependencies([train_gen_a]):
            self.train_gen_a = ema.apply(self.g_A_var)
            
        with tf.control_dependencies([train_disc_a]):
            self.train_disc_a = ema.apply(self.d_A_var)
        
        with tf.control_dependencies([train_gen_b]):
            self.train_gen_b = ema.apply(self.g_B_var)
            
        with tf.control_dependencies([train_disc_b]):
            self.train_disc_b = ema.apply(self.d_B_var)
        
        self.saver = tf.train.Saver()

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        
        #Fake = batch_size of generated items
        #Batch size new = 8
        
        '''
        Temporary storage of fake images + embeddings
        Serves as an experience replay to improve discriminator potency at times
        '''

        if(num_fakes < 64):
            fake_pool[num_fakes:(num_fakes+self.batch_size)] = fake
            return fake
        else:
            rand_idx = np.random.randint(0, 56)
            p = random.random()
            if p > 0.5:
                temp = fake_pool[rand_idx:(rand_idx+self.batch_size)]
                fake_pool[rand_idx:(rand_idx+self.batch_size)] = fake
                return temp
            else:
                return fake
    
    def train(self, data_A, label_A, data_B, epochs, lookup_embed, train_class, continue_train = False):
        
        #data_A = images
        #label_A = corresponding word labels
        #data_B = embeddings
        #lookup_embed = Dictionary for embedding labels
        val = 0
        print("INITIALIZING TRAINING WITH MODEL {}".format(self.name))
        #Data_embed- collection of arbitrary word vectors
        max_len = data_A.shape[0]
        step = 0
        with self.graph.as_default():
        
            if (continue_train):
                self.saver.restore(self.sess, "model/{}.ckpt-6700".format(self.name))
                        
            self.writer = tf.summary.FileWriter("./cycle_logs/{}".format(self.name), self.sess.graph)

            self.sess.run(tf.global_variables_initializer())
            
            done_epoch = False

            for epoch in range(epochs):
                start_idx = 0
                end_idx = self.batch_size
                
                if epoch > 0:
                    #Shuffling consumes too much memory, alter by using this as the main data accessing index
                    perm_idx = np.random.permutation(data_A.shape[0])
                else:
                    perm_idx = np.arange(data_A.shape[0])
                    
                while done_epoch == False:
                                        
                    batch_imgA = []
                    batch_imgB = []
                    
                    for i in range(start_idx, end_idx):
                        noise_matrixA = np.random.normal(0, 1e-3, data_A[perm_idx[i]].shape)
                        noise_matrixB = np.random.normal(0, 1e-3, data_B[0].shape)
                        rand_idx = np.random.randint(0, data_B.shape[0])
                        batch_imgA.append(data_A[perm_idx[i]] + noise_matrixA)
                        batch_imgB.append(data_B[rand_idx] + noise_matrixB)
                    
                    start_idx += self.batch_size
                    end_idx += self.batch_size
                    
                    if (end_idx > max_len):
                        done_epoch = True
                    
                    batch_imgA = np.array(batch_imgA)
                    batch_imgB = np.array(batch_imgB)
                    
                    '''
                    #Reserved for batch size 1
                    if (len(batch_imgA.shape) == 3):
                        batch_imgA = np.expand_dims(batch_imgA, axis = 0)
                    
                    if (len(batch_embedB.shape) == 1):
                        batch_embedB = np.expand_dims(batch_embedB, axis = 0)
                    '''
                    
                    #Training ops and writing to tensorboard
                    
                    #Tensorboard ops
    
                    #Train genB
                    _, fakeB, sum_loss, b_loss, b_grad = self.sess.run([self.train_gen_b, self.fake_B, self.g_loss_bs, self.gen_loss_b, self.var_B],
                                                       feed_dict = {self.img_A: batch_imgA, self.img_B: batch_imgB})
                    
                    self.writer.add_summary(b_grad, step)
                    self.writer.add_summary(sum_loss, step)

                    fakeB_sample = self.fake_image_pool(self.num_fake, fakeB, self.fake_image_B)
                                             
                    #Should be unnecessary
                    if (len(fakeB_sample.shape) == 1):
                        fakeB_sample = np.expand_dims(fakeB_sample, axis = 0)
                    
                    fakeB_sample += noise_matrixB
                    batch_imgB += noise_matrixB
                    
                    #Train discB
                    
                    _, sum_loss, db_grad = self.sess.run([self.train_disc_b, self.d_loss_sum_bs, self.var_Bd],
                                                feed_dict = {self.img_B: batch_imgB, self.fake_B_sample: fakeB_sample})
                    '''    
                    if (val != 0):
                            print("Resuming generator training")
                            val = 0
                    else:
                        if (val == 0):
                            print("Stopping Discriminator Training")
                            val += 1
                        sum_loss, db_grad = self.sess.run([self.d_loss_sum_bs, self.var_Bd], feed_dict = {self.img_B: batch_imgB, self.fake_B_sample: fakeB_sample})
                    '''
                    
                    self.writer.add_summary(db_grad, step)
                    
                    self.writer.add_summary(sum_loss, step)
                    
                    #Train genA
                    _, fakeA, sum_loss, cycle_loss, a_loss, a_grad, var_sanity = self.sess.run([self.train_gen_a, self.fake_A, self.g_loss_as, self.cycle_loss_s, self.gen_loss_a, self.var_A, self.last_layer],
                                                       feed_dict = {self.img_A: batch_imgA, self.img_B: batch_imgB})
                   
                    self.writer.add_summary(var_sanity, step)
                    self.writer.add_summary(a_grad, step)
                    fakeA_sample = self.fake_image_pool(self.num_fake, fakeA, self.fake_image_A)
                    
                    #Should be unnecessary
                    if (len(fakeA_sample.shape) == 3):
                        fakeA_sample = np.expand_dims(fakeA_sample, axis = 0)
                    
                    self.writer.add_summary(sum_loss, step)
                    self.writer.add_summary(cycle_loss, step)
                    
                    #Train discA

                    
                    _, sum_loss, da_grad = self.sess.run([self.train_disc_a, self.d_loss_sum_as, self.var_Ad],
                                                feed_dict = {self.img_A: batch_imgA, self.img_B: batch_imgB, self.fake_A_sample: fakeA_sample})
                    '''
                    else:
                        if (val == 1):
                            print("Stopping Discriminator Training")
                            val += 1
                        sum_loss, da_grad = self.sess.run([self.d_loss_sum_as, self.var_Ad], feed_dict = {self.img_A: batch_imgA, self.fake_A_sample: fakeA_sample})
                    '''
                    self.writer.add_summary(da_grad, step)
    
                    #print(grad2)
                    #print(anotherV)
                    #print('------------------------------')
                    
                    self.writer.add_summary(sum_loss, step)
                    step += 1 
                    self.num_fake += self.batch_size
                    
                    if (step % 5000 == 0 and step != 0):
                        
                        sampleIdx = np.random.randint(0, data_A.shape[0], size = 2)
                        selected_data = data_A[sampleIdx]
                        pred_img = self.sess.run(self.fake_B, feed_dict = {self.img_A: selected_data})
                        plt.imshow(pred_img[0])
                        plt.imshow(pred_img[1])
                        plt.show()
                        
                        #L2 testing for embeddings
                        '''
                        norms = []
                        randIdx = np.random.randint(0, data_A.shape[0], size = 500)
                        corresp_data = data_A[randIdx]
                        corresp_label = label_A[randIdx]
                        fake_emb = []
                        
                        word_label = [cifar10_wlabel(lab, train_class) for lab in corresp_label]
                        embed_label = [lookup_embed[word] for word in word_label]
                        
                        for img, real_embed in zip(corresp_data, embed_label):
                            img = np.expand_dims(img, axis = 0)
                            fake_embed = self.sess.run([self.fake_B], feed_dict = {self.img_A: img})
                            fake_emb.append(fake_embed)
                            norms.append(np.linalg.norm(np.array(real_embed)-np.array(fake_embed))) #Default frobenius norm
                        
                        plt.title("Norms with correct embeddings (l2)")
                        plt.plot(norms)
                        plt.show()
                        
                        optimal_norm = []
                        
                        for fake_embed in fake_emb:
                            min_norm = 1000
                            for word in train_class:
                                trial_norm = np.linalg.norm(np.array(lookup_embed[word]) - np.array(fake_embed))
                                if (trial_norm < min_norm):
                                    min_norm = trial_norm
                            optimal_norm.append(min_norm)
                        
                        plt.title("Optimal Norms")
                        plt.plot(optimal_norm)
                        plt.show()
                        '''
                        print("Saving model at step {}".format(step))
                        print('------------------------------------')
                        self.save(step)
                        
                #Simultaneous Shuffling
                
                #data_A = data_A[perm_idx]
                #label_A = label_A[perm_idx]

                done_epoch = False
                print("Saving model at {} in {}th epoch".format("model/"+self.name+".ckpt", epoch))
                self.save(step)

                print("Finished training {}th epoch".format(epoch))
            
    def save(self, step):
        
        model_path = "model/{}.ckpt".format(self.name)
        self.saver.save(self.sess, model_path, global_step = step)
            
    def load_weights(self, modelName):
        
        self.saver.restore(self.sess, "model/{}".format(modelName))
        
    def predict(self, input_x):
        
        input_x = np.squeeze(input_x)
        
        pred_Img = self.sess.run(self.fake_B, feed_dict = {self.img_A: input_x})
        return pred_Img
    
    def predict_A(self, input_x):
        
        input_x = np.squeeze(input_x)
        
        pred_Img = self.sess.run(self.fake_A, feed_dict = {self.img_B: input_x})
        return pred_Img
