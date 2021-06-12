import tensorflow as tf
import numpy as np
import pandas as pd
import math
import random
import datetime
import os
import time
import matplotlib.pyplot as plt


def weight_variable(shape):
   
    initial = tf.truncated_normal(shape, mean=0, stddev=0.5)
    return tf.Variable(initial)

def weight_variable2(n, shape):
   
    initial = tf.truncated_normal(shape, mean=0, stddev=0.5)
    initial = tf.random_normal(shape, stddev=0.1) 
    
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def add_hiddenLayer2(x, w, b):
    return (tf.tanh(tf.matmul(x, w) + b))

def add_hiddenLayer(x, w, b):
    return (tf.nn.relu(tf.matmul(x, w) + b))

def add_outputLayer(x, w, b=None):
    if b ==  None:
        return (tf.matmul(x, w) )
    else:
        return (tf.matmul(x, w) +b)


import csv

filename = ''   #training data
x_datas = []
y_datas = []
with open(filename) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    first = True
    for row in spamreader:
        if first:
            first = False
            continue
        x_data = row[:-1]
        x_data = [float(f) for f in x_data]
        y_data = row[len(x_data):]
        y_data = [float(f) for f in y_data]
        x_datas.append(x_data)
        y_datas.append(y_data)
x_list=np.array(x_datas)
y_list=np.array(y_datas)


def swap(a,b):
    return b,a
arr[0],arr[1] = swap(arr[0],arr[1])


class SLFN:
    def __init__(self,x_data,y_data):
        self.x_data = np.array(x_data) 
        self.y_data = np.array(y_data)
        x_data = self.x_data.tolist()
        y_data = self.y_data.tolist()
        
        self.x_list=np.array(x_data)
        self.y_list=np.array(y_data)
        
        select_indices = []
        
        type2_index = y_data.index([-1])
        select_indices.append(type2_index)
        type3_index = y_data.index([1])
        select_indices.append(type3_index)
       
        
        for i in range(len(select_indices)):
            
            x_data[i],x_data[select_indices[i]] = swap(x_data[i],x_data[select_indices[i]])
            y_data[i],y_data[select_indices[i]] = swap(y_data[i],y_data[select_indices[i]])
        
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        data_num,_input_size = np.array(x_datas).shape
        
    
        self.inputSize = _input_size 
        self.hiddenNode = 1
        self.outputSize = 1 
        self.eta = 0.1
        self.gamma = 0.1

        self.epsilon_1 = 10e-6
        self.epsilon_2 = 10e-6
        self.v = 0.2
        self.inputx = tf.placeholder(tf.float32, [None, self.inputSize])
        self.inputy = tf.placeholder(tf.float32,[None,self.outputSize])
        self.N = len(self.x_data)
        self.input_eta = tf.placeholder(tf.float32)
        self.input_gamma = tf.placeholder(tf.float32)
        self.reShapeNN2()
        
            
    def Weight_Feedict(self,array):
        feed_dict = {self.w_ih_in:array[0],self.b_ih_in:array[1]
                        ,self.w_ho_in:array[2],self.b_ho_in:array[3]}
        return feed_dict


    def WeightTuning(self,x_data,y_data):
        i=0
        eta=0.1
        gamma=0.1
        
        _weights = self.sess.run(self.weights)
        _weights_feedict = self.Weight_Feedict(_weights)
        
        while (i<100):
            Alpha,Beta,checks = self.calAlphaBeta(x_data,y_data)
    
            input_feed_dict = {self.inputx:x_data, self.inputy:y_data, self.input_eta:eta, self.input_gamma:gamma}
            dis_grad= self.sess.run(self.dis_gra, input_feed_dict)

            if ( False not in checks):
                break
            
            if(dis_grad <= self.epsilon_1):
                break

            else:      
                while True:
                    input_feed_dict = {self.inputx:x_data, self.inputy:y_data, self.input_eta:eta, self.input_gamma:gamma}
                    errorlos,temp_errorlos= self.sess.run((self.errorf,self.temp_errorf), input_feed_dict)
                    if (temp_errorlos <= errorlos):
                        _new_weights = self.sess.run(self.new_weights,input_feed_dict)
                        _new_weights_feed_dict = self.Weight_Feedict(_new_weights)
                        self.sess.run(self.update_weights,feed_dict = _new_weights_feed_dict)
                        eta *= 1.2
                        gamma *=1.2
                        i+=1
                        break
                    else:
                        if (eta > self.epsilon_2):
                            eta *= 0.7
                            gamma *=0.73
                        else:
                            i+=10000
                            break
        alpha,beta,checks = self.calAlphaBeta(x_data,y_data)
        if False in  checks:
            self.sess.run(self.update_weights,_weights_feedict)
            
        if ( i>=100 and i<1000):
            print('--- Weight Tuning 結束 ---')
            
    def LTS_sorting(self,current_index):
        
        trained_x_data =np.array( self.x_data[:current_index])
        trained_y_data =np.array( self.y_data[:current_index])
        untrained_x_data = self.x_data[current_index:]
        untrained_y_data = self.y_data[current_index:]
        input_feed_dict = {self.inputx:untrained_x_data, self.inputy:untrained_y_data, self.input_eta:0.1, self.input_gamma:0.1}
        _error = self.sess.run(self.errorf1,input_feed_dict)
        
        new_indices = np.argsort(_error)
        untrained_x_data = np.array(untrained_x_data)[new_indices]
        untrained_y_data = np.array(untrained_y_data)[new_indices]
        self.x_data = trained_x_data.tolist()+untrained_x_data.tolist()
        self.y_data = trained_y_data.tolist()+untrained_y_data.tolist()
            
    def WeightTuning100(self,x_data,y_data): 
        i=0
        eta=0.1
        gamma=0.1
        _weights = self.sess.run(self.weights)
        _weights_feedict = self.Weight_Feedict(_weights)
        
        while (i<300):
            input_feed_dict = {self.inputx:x_data, self.inputy:y_data, self.input_eta:eta, self.input_gamma:gamma}
            dis_grad= self.sess.run(self.dis_gra, input_feed_dict)
            if(dis_grad <= self.epsilon_1):
                break
            else:
                while True:
                    input_feed_dict = {self.inputx:x_data, self.inputy:y_data, self.input_eta:eta, self.input_gamma:gamma}
                    errorlos,temp_errorlos= self.sess.run((self.errorf,self.temp_errorf), input_feed_dict)
                    if (temp_errorlos <= errorlos):
                        _new_weights = self.sess.run(self.new_weights,input_feed_dict)
                        _new_weights_feed_dict = self.Weight_Feedict(_new_weights)
                        self.sess.run(self.update_weights,feed_dict = _new_weights_feed_dict)
                        eta *= 1.2
                        gamma *=1.2
                        i+=1
                        break
                    else:
                        if (eta > self.epsilon_2):
                            eta *= 0.7
                            gamma *=0.7
                        else:
                            i+=10000000
                            break
        alpha,beta,checks = self.calAlphaBeta(x_data,y_data)
        if False in checks:
            print('weight tuning failed')
            self.sess.run(self.update_weights,_weights_feedict)
       

    def reShapeNN(self):
        self.w_ih = weight_variable([self.inputSize, self.hiddenNode])
        self.b_ih = bias_variable([self.hiddenNode])
        self.hiddenLayer = add_hiddenLayer(self.inputx, self.w_ih, self.b_ih)
        self.w_ho = weight_variable([self.hiddenNode, self.outputSize])
        self.b_ho = bias_variable([self.outputSize])
        
        self.outputLayer = add_outputLayer(self.hiddenLayer, self.w_ho, self.b_ho)
        self.first_y = tf.subtract(self.outputLayer , self.inputy)
        self.errorf1 =tf.reduce_sum(tf.square(self.first_y),axis=1) / (self.N * self.outputSize)
        regular = tf.reduce_sum(tf.square(self.w_ih))+ tf.reduce_sum(tf.square(self.w_ho))
        self.errorf2 = self.errorf1+ ((1e-3/((self.inputSize+1+self.outputSize)*self.hiddenNode)) * regular)
        self.errorf = tf.reduce_sum(self.errorf2)
        self.gra_outw = tf.convert_to_tensor(tf.gradients(self.errorf,self.w_ho))
        self.gra_outbias = tf.convert_to_tensor(tf.gradients(self.errorf,self.b_ho))
        self.gra_hiddenw = tf.convert_to_tensor(tf.gradients(self.errorf,self.w_ih))
        self.gra_hiddenbias = tf.convert_to_tensor(tf.gradients(self.errorf,self.b_ih))
        self.dis_gra = tf.sqrt(tf.add( tf.add( tf.reduce_sum( tf.square( self.gra_outw)),
                                  tf.reduce_sum( tf.square( self.gra_outbias))),
                          tf.add( tf.reduce_sum( tf.square( self.gra_hiddenw)),
                                  tf.reduce_sum( tf.square( self.gra_hiddenbias)))))
        self.w_out = tf.squeeze(tf.subtract(self.w_ho , tf.multiply(self.input_eta , tf.divide(self.gra_outw,self.dis_gra))),0)
        self.b_out = tf.squeeze(tf.subtract(self.b_ho , tf.multiply(self.input_eta , tf.divide(self.gra_outbias,self.dis_gra))),0)
        self.w_hidd = tf.squeeze(tf.subtract(self.w_ih , tf.multiply(self.input_eta , tf.divide(self.gra_hiddenw,self.dis_gra))),0)
        self.b_hidd = tf.squeeze(tf.subtract(self.b_ih , tf.multiply(self.input_eta , tf.divide(self.gra_hiddenbias,self.dis_gra))),0)
        
        self.temp_hiddenLayer = add_hiddenLayer(self.inputx, self.w_hidd, self.b_hidd)
        
        self.temp_outputLayer = add_outputLayer(self.temp_hiddenLayer, self.w_out, self.b_out) 
        self.temp_y = tf.subtract( self.temp_outputLayer,self.inputy)
        self.temp_errorf1 =tf.reduce_sum(tf.square(self.temp_y)) / (self.N * self.outputSize)
        regular2 = tf.reduce_sum(tf.square(self.w_out)) + tf.reduce_sum(tf.square(self.w_hidd))
        regular2 += tf.reduce_sum(tf.square(self.b_out)) + tf.reduce_sum(tf.square(self.b_hidd))
        self.temp_errorf =self.temp_errorf1 + ((1e-3/((self.inputSize+1+self.outputSize)*self.hiddenNode)) * regular2)
        
        self.w_ho_in =tf.placeholder(tf.float32,self.w_ho.shape,name="w_ho_in")
        self.b_ho_in =tf.placeholder(tf.float32,self.b_ho.shape,name="b_ho_in")
        self.w_ih_in =tf.placeholder(tf.float32,self.w_ih.shape,name="w_ih_in")
        self.b_ih_in =tf.placeholder(tf.float32,self.b_ih.shape,name="b_ih_in")
        self.update_w_ho = tf.assign(self.w_ho,self.w_ho_in)
        self.update_b_ho = tf.assign(self.b_ho,self.b_ho_in)
        self.update_w_ih = tf.assign(self.w_ih,self.w_ih_in)
        self.update_b_ih = tf.assign(self.b_ih,self.b_ih_in)
        self.update_weights = [self.update_w_ih,self.update_b_ih,self.update_w_ho,self.update_b_ho]
        self.new_weights = [self.w_hidd,self.b_hidd,self.w_out,self.b_out]
        self.weights = [self.w_ih,self.b_ih,self.w_ho,self.b_ho]
        self.new_weights_feed_dict =self.Weight_Feedict(self.new_weights)
        self.sess=tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)     
    def reShapeNN2(self):
        self.w_ih = weight_variable2(0, [self.inputSize, self.hiddenNode])
        
        self.b_ih = weight_variable2(1, [self.hiddenNode])
        self.hiddenLayer = add_hiddenLayer(self.inputx, self.w_ih, self.b_ih)
        self.w_ho = weight_variable2(2, [self.hiddenNode, self.outputSize])
        self.b_ho = weight_variable2(3, [self.outputSize])
        
        self.outputLayer = add_outputLayer(self.hiddenLayer, self.w_ho, self.b_ho)
        self.first_y = tf.subtract(self.outputLayer , self.inputy)
        self.errorf1 =tf.reduce_sum(tf.square(self.first_y),axis=1) / (self.N * self.outputSize)
        regular = tf.reduce_sum(tf.square(self.w_ih))+ tf.reduce_sum(tf.square(self.w_ho))

        self.errorf2 = self.errorf1+ ((1e-3/((self.inputSize+1+self.outputSize)*self.hiddenNode)) * regular)
        self.errorf = tf.reduce_sum(self.errorf2)
        self.gra_outw = tf.convert_to_tensor(tf.gradients(self.errorf,self.w_ho))
        self.gra_outbias = tf.convert_to_tensor(tf.gradients(self.errorf,self.b_ho))
        self.gra_hiddenw = tf.convert_to_tensor(tf.gradients(self.errorf,self.w_ih))
        self.gra_hiddenbias = tf.convert_to_tensor(tf.gradients(self.errorf,self.b_ih))
        self.dis_gra = tf.sqrt(tf.add( tf.add( tf.reduce_sum( tf.square( self.gra_outw)),
                                  tf.reduce_sum( tf.square( self.gra_outbias))),
                          tf.add( tf.reduce_sum( tf.square( self.gra_hiddenw)),
                                  tf.reduce_sum( tf.square( self.gra_hiddenbias)))))
        self.w_out = tf.squeeze(tf.subtract(self.w_ho , tf.multiply(self.input_eta , tf.divide(self.gra_outw,self.dis_gra))),0)
        self.b_out = tf.squeeze(tf.subtract(self.b_ho , tf.multiply(self.input_eta , tf.divide(self.gra_outbias,self.dis_gra))),0)
        self.w_hidd = tf.squeeze(tf.subtract(self.w_ih , tf.multiply(self.input_eta , tf.divide(self.gra_hiddenw,self.dis_gra))),0)
        self.b_hidd = tf.squeeze(tf.subtract(self.b_ih , tf.multiply(self.input_eta , tf.divide(self.gra_hiddenbias,self.dis_gra))),0)
        
        self.temp_hiddenLayer = add_hiddenLayer(self.inputx, self.w_hidd, self.b_hidd)
       
        self.temp_outputLayer = add_outputLayer(self.temp_hiddenLayer, self.w_out, self.b_out)   
        
        self.temp_y = tf.subtract( self.temp_outputLayer,self.inputy)
        
        self.temp_errorf1 =tf.reduce_sum(tf.square(self.temp_y)) / (self.N * self.outputSize)
        regular2 = tf.reduce_sum(tf.square(self.w_out)) + tf.reduce_sum(tf.square(self.w_hidd))
        regular2 += tf.reduce_sum(tf.square(self.b_out)) + tf.reduce_sum(tf.square(self.b_hidd))
        self.temp_errorf =self.temp_errorf1 + ((1e-3/((self.inputSize+1+self.outputSize)*self.hiddenNode)) * regular2)
        
        self.w_ho_in =tf.placeholder(tf.float32,self.w_ho.shape,name="w_ho_in")
        self.b_ho_in =tf.placeholder(tf.float32,self.b_ho.shape,name="b_ho_in")
        self.w_ih_in =tf.placeholder(tf.float32,self.w_ih.shape,name="w_ih_in")
        self.b_ih_in =tf.placeholder(tf.float32,self.b_ih.shape,name="b_ih_in")
        
        self.update_w_ho = tf.assign(self.w_ho,self.w_ho_in)
        self.update_b_ho = tf.assign(self.b_ho,self.b_ho_in)
        self.update_w_ih = tf.assign(self.w_ih,self.w_ih_in)
        self.update_b_ih = tf.assign(self.b_ih,self.b_ih_in)
        
        self.update_weights = [self.update_w_ih,self.update_b_ih,self.update_w_ho,self.update_b_ho]
        self.new_weights = [self.w_hidd,self.b_hidd,self.w_out,self.b_out]
        self.weights = [self.w_ih,self.b_ih,self.w_ho,self.b_ho]
        self.new_weights_feed_dict =self.Weight_Feedict(self.new_weights)
        
        self.sess=tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)     
        
        
        
    def calAlphaBeta(self,x_data,y_data,output_index=-1):
        if output_index == -1:
            checks = [False]
            Alphas = []
            Betas = []
            vals = []
            for i in range(self.outputSize):
                
                alpha_index,beta_index = -1,-1
                
                Alpha = 10e6
                Beta = -10e6
                for j in range(len(x_data)):
                    val = self.sess.run(self.outputLayer, feed_dict = {self.inputx:[x_data[j]]})
                    val = val[0][i]
                    
                    if y_data[j][i] == 1:
                        if val < Alpha:
                            Alpha = val
                            alpha_index = j
                    else :
                        if val > Beta:
                            Beta = val
                            beta_index = j
                 
                Alphas.append(Alpha)
                Betas.append(Beta)
                checks[i] =  Alpha>Beta
            return Alphas,Betas,checks
        Alpha = 10e6
        Beta = -10e6
        for i in range(0,len(x_data)):
            val = self.sess.run(self.outputLayer, feed_dict = {self.inputx:[x_data[i]]})
            val = val[0][output_index]
            if y_data[i][output_index] == 1:
                if val < Alpha:
                    Alpha = val
            else:
                if val > Beta:
                    Beta = val
        return Alpha,Beta, Alpha>Beta
   
    
        
        
    def Cramming(self,x_data,y_data,Last_Alpha,Last_Beta,output_index):
        _new_weight = self.sess.run(self.weights)
        _new_weight = [f.tolist() for f in _new_weight]
        index = len(x_data)-1
        val = self.sess.run(self.outputLayer[0][output_index],feed_dict={self.inputx:[x_data[index]]})
        
        if y_data[index][output_index] == 1:#
            
            _ho_w = np.subtract(Last_Beta ,val)
        else:
            
            _ho_w =np.subtract( Last_Alpha , val)
            
       
        
        xc = self.x_data[0:(self.n-1)]
        
        xn = self.x_data[self.n-1]
        
        
        alpha = np.zeros((self.inputSize, 1))
        while(1):
            alpha = np.random.rand(self.inputSize, 1)
            alpha = tf.cast(alpha,tf.float32)
            
            alphaLen = tf.sqrt(tf.reduce_sum(tf.square(alpha)))
            alpha = tf.divide(alpha, alphaLen)
            alphaCBarVal = self.sess.run(tf.matmul(tf.subtract(xc,xn), alpha))
            
            case1 = (np.count_nonzero(alphaCBarVal) == self.n-1)
            if(case1 == True):
                alphaCBarVal = min(abs(alphaCBarVal))[0]
                break
        
        zeta = tf.constant(0.1, tf.float32)
        diff = self.sess.run((tf.cast((np.array(xc)-np.array(xn)), tf.float32)))
        
        while(1):
            para1 = zeta + tf.matmul(tf.transpose(alpha), diff.reshape(self.inputSize, self.n - 1))
            
            para2 = zeta - tf.matmul(tf.transpose(alpha), diff.reshape(self.inputSize, self.n - 1))
            
            case2 = self.sess.run(para1 * para2) < 0
            if(case2.all() == True):
                
                break
            else:
                zeta = zeta * 0.5
                
                
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        
        
        wih0 = (1 * alpha).eval(session=sess) 
        
        for i in range(self.inputSize):
            _new_weight[0][i].extend(wih0[i])
            _new_weight[0][i].extend(wih0[i])
            _new_weight[0][i].extend(wih0[i])
            _new_weight[0][i].extend(wih0[i])
            
        
        
        _xn = self.sess.run((tf.cast(np.array(xn), tf.float32)))
        bih0 = (tf.reshape((-1*zeta) - 1 * tf.matmul(tf.transpose(alpha), _xn.reshape(self.inputSize, 1)), [1])).eval(session=sess)
        bih1 = (tf.reshape(- 1 * tf.matmul(tf.transpose(alpha), _xn.reshape(self.inputSize, 1)), [1])).eval(session=sess)
        bih3 = (tf.reshape(zeta - 1 * tf.matmul(tf.transpose(alpha), _xn.reshape(self.inputSize, 1)), [1])).eval(session=sess)  
        
        _new_weight[1].extend(bih0)
        _new_weight[1].extend(bih1)
        _new_weight[1].extend(bih1)
        _new_weight[1].extend(bih3)
        if y_data[index][output_index] == 1:
            who = self.sess.run((np.subtract(Last_Beta ,val)/zeta)*1.5)
            
        elif y_data[index][output_index] == -1:
            who = self.sess.run((np.subtract(Last_Alpha ,val)/zeta)*1.5)
        _arr = []
        for i in range(self.outputSize):
            _arr.append(who)
            _arr.append(-1*who)
            _arr.append(-1*who)
            _arr.append(who)
            
        ar1 = [[v] for v in _arr]
             
        _new_weight[2].extend(ar1)
        self.hiddenNode = self.hiddenNode + 4
        self.reShapeNN()
        
        
        _new_weight_feedict = self.Weight_Feedict(_new_weight)
        
        
        self.sess.run(self.update_weights,_new_weight_feedict)
        _weight = self.sess.run(self.weights)
        val2 = self.sess.run(self.outputLayer[0][output_index],feed_dict={self.inputx:[x_data[index]]})
        
        
        alpha,beta,check = self.calAlphaBeta(x_data,y_data,output_index)
  
        
    def PruningHiddenNode(self,KillIndex):
        assert KillIndex >= 0 and KillIndex < self.hiddenNode
        _weights = self.sess.run(self.weights)
        _weights = np.array(_weights)
        
        _new_w_ih = []
        for i in range(len(_weights[0])):
            _new_weight =  _weights[0][i][0:KillIndex].tolist()  + _weights[0][i][KillIndex+1:].tolist() 
            _new_w_ih.append(_new_weight)

        _new_b_ih = _weights[1][0:KillIndex].tolist() + _weights[1][KillIndex+1:].tolist() 
        #2,2 => 1,2
        arr = []
        for i in range(self.hiddenNode):
            if i != KillIndex:
                _new_w_ho = _weights[2][i].tolist() 
                arr.append(_new_w_ho)
        _new_w_ho = arr
        
        _new_b_ho = _weights[3]
        _new_weight = [_new_w_ih,_new_b_ih,_new_w_ho,_new_b_ho]

        return  _new_weight
    
    def pruning(self,KillIndex):
        if self.hiddenNode == 1:
            
            return False
        
        _new_weight = self.PruningHiddenNode(KillIndex)
        
        self.hiddenNode = self.hiddenNode-1
        self.reShapeNN()
        _new_weight_feedict = self.Weight_Feedict(_new_weight)
        self.sess.run(self.update_weights,_new_weight_feedict)
        return True

    def TempPruningHiddenNode(self,KillIndex):
        
        assert KillIndex >= 0 and KillIndex < self.hiddenNode

        _weights = self.sess.run(self.weights)
        _weights = np.array(_weights)

        _new_w_ih  = _weights[0]
        _new_b_ih = _weights[1][0:KillIndex].tolist() +[0]+ _weights[1][KillIndex+1:].tolist() 
        _new_w_ho = _weights[2][0:KillIndex].tolist()+[[0 for f in range(self.outputSize)]]+ _weights[2][KillIndex+1:].tolist() 
        _new_b_ho = _weights[3]
        _new_weight = [_new_w_ih,_new_b_ih,_new_w_ho,_new_b_ho]
        _new_weight_feedict = self.Weight_Feedict(_new_weight)
        return _new_weight_feedict
    
    def distance_v(self):
        Alhpas,Betas,checks = calAlphaBeta(self.x_data,self.y_data)
        _weights=self.sess.run(self.weights)
        
        for i in range(self.outputSize):
            alpha = Alphas[i]
            beta = Betas[i]
            
            v_temp = (2*self.v) / (alpha-beta)
            
            
            _ho_w = _weights[2]
            #[1,2]
            _ho_w[0][i] = v_temp*f
            
            _weights[3][i]= self.v-2*self.v*(alpha-b)/ (alpha-beta)
            
            #_weights[3][i]=  self.v-alpha 
            
        _Weight_Feedict = self.Weight_Feedict(_weights)
        self.sess.run(self.update_weights,_Weight_Feedict)
        return True
    def calgks(self,x_data,y_data):
        gks = []
        for i in range(self.outputSize):
            Alpha,Beta,_=self.calAlphaBeta(x_data,y_data,i)

            gks.append( Alpha - Beta )

        conditions = []

        for gk in gks:
            if gk >0:
                j = 1
            elif gk <= 0 and gk >=-self.theta:
                j = 2
            else:
                j = 3
            conditions.append(j)
        return gks,conditions
        
    def TrainWithSLFN(self):#開始訓練
        
        tStart = time.time()
        
        
        Last_Alphas = [0 for f in range(self.outputSize)]
        Last_Betas = [0 for f in range(self.outputSize)]
        Alphas = []
        Betas = []
        
        index_list=[]
        hidden_num=[]
        
        assert len(self.x_data) >=2
        
        for i in range(self.outputSize):
            Alpha,Beta,check=self.calAlphaBeta(self.x_data[:2],self.y_data[:2],i)
            if not check:
                _weights = self.sess.run(self.weights)
                _weights[2][0][i] = -_weights[2][0][i]
                
                _Weight_Feedict = self.Weight_Feedict(_weights)
                
                self.sess.run(self.update_weights,_Weight_Feedict)
                Alpha,Beta,check=self.calAlphaBeta(self.x_data[0:2],self.y_data[:2],i)
                
                
            Alphas.append(Alpha)
            Last_Alphas[i] = Alpha
            
            Betas.append(Beta)
            Last_Betas[i] = Beta
            
        Alpha,Beta,checks =self.calAlphaBeta(self.x_data[:2],self.y_data[:2])
       

        self.n = 2
        N=len(self.x_data)
        self.theta =1
        
        while(True):
            
            
            
            assert len(Last_Alphas) == len(Last_Betas) == 1
            self.n += 1 
            
            if self.n> (N*0.95):
                break
            self.LTS_sorting(self.n-1)    
            _x_data = self.x_data[:self.n]
            _y_data = self.y_data[:self.n]
            
           
            x_index = x_datas.index(_x_data[-1])
            index_list.append(x_index)
            
            Alpha,Beta,checks = self.calAlphaBeta(_x_data,_y_data)
            if False in checks:
                #WeghtTuning
                self.WeightTuning(_x_data,_y_data)
                
            for i in range(self.outputSize):
                _,_,checks =self.calAlphaBeta(_x_data,_y_data)
                Last_Alphas,Last_Betas,_ =self.calAlphaBeta(_x_data[:-1],_y_data[:-1])
                if checks[i] == True:
                    continue
                #print('Craming node:%d'%i)
                
                self.Cramming(_x_data,_y_data,Last_Alphas[i],Last_Betas[i],i)
            _hiddenNode = -1
            
            _weights = self.sess.run(self.weights)
            _Weight_Feedict = self.Weight_Feedict(_weights)
            
            while(True):
                
            
                if (self.n % 10 !=0):
                    break
                
                _hiddenNode += 1 
                if _hiddenNode >= self.hiddenNode or self.hiddenNode == 1:
                    break
                
                _Pruning_Weight_feedict = self.TempPruningHiddenNode(_hiddenNode)
                self.sess.run(self.update_weights,feed_dict=_Pruning_Weight_feedict)

                gks,conditions = self.calgks(_x_data,_y_data)
                
               
                
                self.sess.run(self.update_weights,feed_dict=_Weight_Feedict)

                if [1] == conditions :
                    if self.hiddenNode==1:
                        
                        break
                    check = self.pruning(_hiddenNode)
                    _hiddenNode -=1
                    _weights = self.sess.run(self.weights)
                    _Weight_Feedict = self.Weight_Feedict(_weights)
                    
                    self.WeightTuning100(_x_data,_y_data)
                    
                    continue
                    
                if  3 in conditions:
                    
                    continue

                if (1 in conditions and 2 in conditions ) or ([2] == conditions) :
                    if self.hiddenNode==1:
                        
                        break
                    
                    check = self.pruning(_hiddenNode)
                    _hiddenNode -=1

                    self.WeightTuning100(_x_data,_y_data)
                    gks,conditions = self.calgks(_x_data,_y_data)

                    if [1] == conditions:
                        
                        _weights = self.sess.run(self.weights)
                        _Weight_Feedict = self.Weight_Feedict(_weights)
                        self.WeightTuning100(_x_data,_y_data)
                        continue
                    else:
                        self.hiddenNode+=1
                        _hiddenNode +=1
                        
                        self.reShapeNN()
                      
                        _Weight_Feedict = self.Weight_Feedict(_weights)
                        self.sess.run(self.update_weights,feed_dict=_Weight_Feedict)
                        continue
                
                
               
            Last_Alphas,Last_Betas,_=self.calAlphaBeta(_x_data,_y_data)
            
            current = time.time()
            hidden_num.append(self.hiddenNode)
            
            
            _weights = self.sess.run(self.weights)
            
            
            y_diff = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.inputy, self.outputLayer))))
            sigma = (self.sess.run(y_diff,feed_dict={self.inputx:self.x_list[:self.n], self.inputy:self.y_list[:self.n]}))/self.n
            
            


_SLFN = SLFN(x_datas,y_datas)
starttime = datetime.datetime.now()
_SLFN.TrainWithSLFN()

