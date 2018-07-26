
#CycleGAN Implementation

#Imports
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from math import floor


#Noise And Noisy Labels
def zero():
    return np.random.uniform(0.0, 0.01, size = [1])

def one():
    return np.random.uniform(0.99, 1.0, size = [1])

def noise(n):
    return np.random.uniform(-1.0, 1.0, size = [n, 2048])


#Import Images
print("Importing Images...")

ImagesA = []
ImagesB = []
n_images = 1068

for n in range(1, n_images):
    tempA = Image.open("data/DomainA/im ("+str(n)+").png")
    tempB = Image.open("data/DomainB/im ("+str(n)+").png")
    
    tempA1 = np.array(tempA.convert('RGB'), dtype='float32')
    tempB1 = np.array(tempB.convert('RGB'), dtype='float32')
    
    ImagesA.append(tempA1 / 255)
    ImagesB.append(tempB1 / 255)
    
    ImagesA.append(np.flip(ImagesA[-1], 1))
    ImagesB.append(np.flip(ImagesB[-1], 1))




#Keras Imports
from keras.models import Sequential, model_from_json, Model
from keras.layers import Conv2D, LeakyReLU, AveragePooling2D, BatchNormalization, Reshape, Dense
from keras.layers import UpSampling2D, Activation, Dropout, concatenate, Input, Flatten
from keras.optimizers import Adam
import keras.backend as K


#Defining Layers For U-Net
def conv(input_tensor, filters, bn = True, drop = 0):
    
    co = Conv2D(filters = filters, kernel_size = 3, padding = 'same')(input_tensor)
    ac = LeakyReLU(0.2)(co)
    ap = AveragePooling2D()(ac)
    
    if bn:
        ap = BatchNormalization(momentum = 0.75)(ap)
        
    if drop > 0:
        ap = Dropout(drop)(ap)
    
    return ap

def deconv(input1, input2, filters, drop = 0):
    #Input 1 Should be half the size of Input 2
    up = UpSampling2D()(input1)
    co = Conv2D(filters = filters, kernel_size = 3, padding = 'same')(up)
    ac = Activation('relu')(co)
    
    if drop > 0:
        ac = Dropout(drop)(ac)
        
    ba = BatchNormalization(momentum = 0.75)(ac)
    con = concatenate([ba, input2])
    
    return con


#Smooth L1 Loss Function
HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
   x   = K.abs(y_true - y_pred)
   x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.sum(x)


#Define The Actual Model Class
class GAN(object):
    
    def __init__(self):
        
        #Always 256x256 Images
        
        #Models
        
        #Generator (Domain A -> Domain B)
        self.G1 = None
        
        #Generator (Domain B -> Domain A)
        self.G2 = None
        
        #Discriminator (Domain B)
        self.D = None
        
        #Old Discriminator For Rollback After Training Generator
        self.OD = None
        
        #Training Models
        self.DM = None #Discriminator Model (D)
        self.AM = None #Aversarial Model (G1 + D)
        self.TM = None #Together Model (G1 + G2)
        
        
        #Other Config
        self.LR = 0.00015 #Learning Rate
        self.steps = 1 #Training Steps Taken
    
    def generator1(self):
        
        #Defining G1 // U-Net
        if self.G1:
            return self.G1
        
        #Image Input
        inp_i = Input(shape = [256, 256, 3])
        
        #Noise Input
        inp_n = Input(shape = [2048])
        
        #256
        d0 = conv(inp_i, 8, False)
        #128
        d1 = conv(d0, 16)
        #64
        d2 = conv(d1, 32)
        #32
        d3 = conv(d2, 64)
        #16
        d4 = conv(d3, 128)
        #8
        d5 = conv(d4, 256)
        #4
        
        center = Conv2D(filters = 256, kernel_size = 3, padding = 'same')(d5)
        ac = LeakyReLU(0.2)(center)
        
        #Add Noise
        rs = Reshape(target_shape = [4, 4, 128])(inp_n)
        center2 = concatenate([rs, ac])
        
        #4
        u0 = deconv(center2, d4, 256)
        #8
        u1 = deconv(u0, d3, 128)
        #16
        u2 = deconv(u1, d2, 64)
        #32
        u3 = deconv(u2, d1, 32)
        #64
        u4 = deconv(u3, d0, 16)
        #128
        u5 = UpSampling2D()(u4)
        cc = concatenate([inp_i, u5])
        cl = Conv2D(filters = 8, kernel_size = 3, padding = 'same', activation = 'relu')(cc)
        #256
        out = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(cl)
        
        self.G1 = Model(inputs = [inp_i, inp_n], outputs = out)
        
        return self.G1
    
    def generator2(self):
        
        #Defining G2 // U-Net
        if self.G2:
            return self.G2
        
        #Image Input
        inp_i = Input(shape = [256, 256, 3])
        
        #256
        d0 = conv(inp_i, 8, False)
        #128
        d1 = conv(d0, 16)
        #64
        d2 = conv(d1, 32)
        #32
        d3 = conv(d2, 64)
        #16
        d4 = conv(d3, 128)
        #8
        d5 = conv(d4, 256)
        #4
        
        center = Conv2D(filters = 256, kernel_size = 3, padding = 'same')(d5)
        ac = LeakyReLU(0.2)(center)
        
        #No Noise
        
        #4
        u0 = deconv(ac, d4, 256)
        #8
        u1 = deconv(u0, d3, 128)
        #16
        u2 = deconv(u1, d2, 64)
        #32
        u3 = deconv(u2, d1, 32)
        #64
        u4 = deconv(u3, d0, 16)
        #128
        u5 = UpSampling2D()(u4)
        cc = concatenate([inp_i, u5])
        cl = Conv2D(filters = 8, kernel_size = 3, padding = 'same', activation = 'relu')(cc)
        #256
        out = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(cl)
        
        self.G2 = Model(inputs = inp_i, outputs = out)
        
        return self.G2
    
    def discriminator(self):
        
        #Defining D
        if self.D:
            return self.D
        
        #Image Input
        inp_i = Input(shape = [256, 256, 3])
        
        #256
        d0 = conv(inp_i, 8, False, drop = 0.2)
        #128
        d1 = conv(d0, 16, drop = 0.2)
        #64
        d2 = conv(d1, 32, drop = 0.2)
        #32
        d3 = conv(d2, 64, drop = 0.2)
        #16
        d4 = conv(d3, 128, drop = 0.2)
        #8
        d5 = conv(d4, 256, drop = 0.2)
        #4
        fl = Flatten()(d5)
        #4096 (Flat)
        d1 = LeakyReLU(0.2)(Dense(256)(fl))
        #512
        out = Dense(1, activation = 'sigmoid')(d1)
        #Binary Output
        
        self.D = Model(inputs = inp_i, outputs = out)
        
        return self.D
    
    def DisModel(self):
        
        #Defining DM
        if self.DM == None:
            self.DM = Sequential()
            self.DM.add(self.discriminator())
        
        # Incrementally Dropping LR
        # self.LR * (0.9 ** floor(self.steps / 10000))
        self.DM.compile(optimizer = Adam(lr = self.LR * (0.9 ** floor(self.steps / 10000))),
                        loss = 'binary_crossentropy')
        
        return self.DM
    
    def AdModel(self):
        
        #Defining AM
        if self.AM == None:
            #Image Input
            in1 = Input(shape = [256, 256, 3])
            #Noise Input
            in2 = Input(shape = [2048])
            #G1 Part
            g1 = self.generator1()([in1, in2])
            #D Part
            d = self.discriminator()(g1)
            
            self.AM = Model(inputs = [in1, in2], outputs = d)
        
        # Incrementally Dropping LR
        # self.LR * (0.9 ** floor(self.steps / 10000))
        self.AM.compile(optimizer = Adam(lr = self.LR * (0.9 ** floor(self.steps / 10000))),
                        loss = 'binary_crossentropy')
        
        return self.AM
    
    def TogModel(self):
        
        #Defining TM
        if self.TM == None:
            #Image Input
            in1 = Input(shape = [256, 256, 3])
            #Noise Input
            in2 = Input(shape = [2048])
            #G1 Part
            g1 = self.generator1()([in1, in2])
            #G2 Part
            g2 = self.generator2()(g1)
            
            self.TM = Model(inputs = [in1, in2], outputs = g2)
        
        # Incrementally Dropping LR
        # self.LR * (0.9 ** floor(self.steps / 10000))
        self.TM.compile(optimizer = Adam(lr = self.LR * 0.5 * (0.9 ** floor(self.steps / 10000))),
                        loss = smoothL1)
        
        return self.TM
    
    def sod(self):
        
        #Save Old Discriminator
        self.OD = self.D.get_weights()
    
    def lod(self):
        
        #Load Old Discriminator
        self.D.set_weights(self.OD)



#Now Define The Actual Model
class CycleGAN(object):
    
    def __init__(self, steps = -1):
        
        #Models
        #Main
        self.GAN = GAN()
        
        #Set Steps, If Relevant
        if steps >= 0:
            self.GAN.steps = steps
        
        #Generators
        self.G1 = self.GAN.generator1()
        self.G2 = self.GAN.generator2()
        
        #Training Models
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.TogModel = self.GAN.TogModel()
        
    def train(self, batch = 16):
        
        (a, b) = self.train_dis(batch)
        c = self.train_ad(batch)
        d = self.train_tog(batch)
        
        
        print("D Real: " + str(a))
        print("D Fake: " + str(b))
        print("G1 D::: " + str(c))
        print("G1 G2:: " + str(d))
        
        if self.GAN.steps % 50 == 0:
            self.save(floor(self.GAN.steps / 1000))
            self.evaluate()
            
        if self.GAN.steps % 10000 == 0:
            self.GAN.AM = None
            self.GAN.DM = None
            self.GAN.TM = None
            self.AdModel = self.GAN.AdModel()
            self.DisModel = self.GAN.DisModel()
            self.TogModel = self.GAN.TogModel()
        
        self.GAN.steps = self.GAN.steps + 1
        
        return True
    
    def train_dis(self, batch):
        
        number = max(1, int(batch / 2))
        
        #Get Real Images
        train_data = []
        label_data = []
        for i in range(number):
            im_no = random.randint(0, len(ImagesB) - 1)
            train_data.append(ImagesB[im_no])
            label_data.append(one())
            
        d_loss_real = self.DisModel.train_on_batch(np.array(train_data), np.array(label_data))
        
        #Get Fake Images
        train_data = []
        label_data = []
        
        for i in range(number):
            im_no = random.randint(0, len(ImagesA) - 1)
            label_data.append(zero())
            
            if random.random() < 0.6:
                train_data.append(self.G1.predict([np.array([ImagesA[im_no]]), noise(1)])[0])
            else:
                train_data.append(ImagesB[im_no])
            
        d_loss_fake = self.DisModel.train_on_batch(np.array(train_data), np.array(label_data))
        
        del train_data, label_data
        
        return (d_loss_real, d_loss_fake)
    
    def train_ad(self, batch):
        
        #Save Old Discriminator
        self.GAN.sod()
        
        #Labels And Train Images
        label_data = []
        train_data = []
        for i in range(int(batch)):
            im_no = random.randint(0, len(ImagesA) - 1)
            train_data.append(ImagesA[im_no])
            label_data.append(one())
        
        g_loss = self.AdModel.train_on_batch([np.array(train_data), noise(batch)], np.array(label_data))
        
        #Load Old Discriminator
        self.GAN.lod()
        
        del train_data, label_data
        
        return g_loss
    
    def train_tog(self, batch):
        
        #Labels And Train Images
        label_data = []
        train_data = []
        for i in range(int(batch)):
            im_no = random.randint(0, len(ImagesA) - 1)
            train_data.append(ImagesA[im_no])
            label_data.append(ImagesA[im_no])
        
        g_loss = self.TogModel.train_on_batch([np.array(train_data), noise(batch)], np.array(label_data))
        
        del train_data, label_data
        
        return g_loss
        
    def evaluate(self, show = True):
        
        im_no = random.randint(0, len(ImagesA) - 1)
        im1 = ImagesA[im_no]
        im2 = ImagesB[im_no]
        
        im3 = self.G1.predict([np.array([im1]), noise(1)])
        
        im4 = self.G2.predict(im3)
        
        im5 = np.concatenate([im1, im2], axis = 1)
        im6 = np.concatenate([im3[0], im4[0]], axis = 1)
        
        im7 = np.concatenate([im5, im6], axis = 0)
        
        if show:
            plt.figure(1)
            plt.imshow(im5)
            plt.figure(2)
            plt.imshow(im6)
            plt.show()
            
        del im1, im2, im3, im4, im5, im6
        
        return im7
    
    def eval2(self, num):
        
        im = []
        
        #Blank Space
        brow = np.zeros(shape = [16, 2096, 3])
        bcol = np.zeros(shape = [512, 16, 3])
        
        #Get 12 Tries
        for _ in range(12):
            im.append(self.evaluate(False))
            im.append(bcol)
        
        #Concatenate Rows
        row1 = np.concatenate(im[:7], axis = 1)
        row2 = np.concatenate(im[8:15], axis = 1)
        row3 = np.concatenate(im[16:23], axis = 1)
        
        image = np.concatenate([row1, brow, row2, brow, row3], axis = 0)
        
        x = Image.fromarray(np.uint8(image*255))
        
        x.save("Results/i"+str(num)+".png")
        
        del row1, row2, row3, image, x
            
    
    def save(self, num):
        gen1_json = self.GAN.G1.to_json()
        gen2_json = self.GAN.G2.to_json()
        dis_json = self.GAN.D.to_json()

        with open("Models/gen1.json", "w") as json_file:
            json_file.write(gen1_json)
        
        with open("Models/gen2.json", "w") as json_file:
            json_file.write(gen2_json)

        with open("Models/dis.json", "w") as json_file:
            json_file.write(dis_json)

        self.GAN.G1.save_weights("Models/gen1_"+str(num)+".h5")
        self.GAN.G2.save_weights("Models/gen2_"+str(num)+".h5")
        self.GAN.D.save_weights("Models/dis"+str(num)+".h5")

        #print("Saved!")

    def load(self, num):
        steps1 = self.GAN.steps
        
        self.GAN = None
        self.GAN = GAN()

        #Generator 1
        gen_file = open("Models/gen1.json", 'r')
        gen_json = gen_file.read()
        gen_file.close()
        
        self.GAN.G1 = model_from_json(gen_json)
        self.GAN.G1.load_weights("Models/gen1_"+str(num)+".h5")
        
        #Generator 2
        gen_file = open("Models/gen2.json", 'r')
        gen_json = gen_file.read()
        gen_file.close()
        
        self.GAN.G2 = model_from_json(gen_json)
        self.GAN.G2.load_weights("Models/gen2_"+str(num)+".h5")

        #Discriminator
        dis_file = open("Models/dis.json", 'r')
        dis_json = dis_file.read()
        dis_file.close()
        
        self.GAN.D = model_from_json(dis_json)
        self.GAN.D.load_weights("Models/dis"+str(num)+".h5")
        
        self.GAN.steps = steps1

        #Reinitialize
        self.G1 = self.GAN.generator1()
        self.G2 = self.GAN.generator2()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.TogModel = self.GAN.TogModel()

#Finally Onto The Main Function
model = CycleGAN()

while(True):
    print("\n\nRound " + str(model.GAN.steps) + ":")
    model.train(1)
    
    if model.GAN.steps % 1000 == 0:
        model.eval2(floor(model.GAN.steps / 1000))
    
    


























