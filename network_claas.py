#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import scipy
# neural network class definition
import scipy.special
import numpy 

# library for plotting arrays
import matplotlib.pyplot
# helper to load data from PNG image files# helpe 
import imageio
# glob helps select multiple files using patterns
import glob

class neuralNetwork : 
    
    # initialise the neural network 
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate) : 
        
        #set number of nodes in each input , hidden , output
        
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who 
        # weights inside the arrays are w_ i_ j, where link is from node i to node j in the next layer 
        # w11 w21 
        # w12 w22 etc 
       
        self. wih = numpy.random.normal( 0.0, pow( self. hnodes, -0.5), (self. hnodes, self. inodes))
        self. who = numpy.random.normal( 0.0, pow( self. onodes, -0.5), (self. onodes, self. hnodes))

        # learning rate 
        self.lr = learningrate

        # activation function is the sigmoid function
       
        self.activation_function = lambda x:scipy.special.expit(x) 

        pass 
    
    # train the neural network 
    def train(self, inputs_list, targets_list) : 


        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        

        hidden_inputs = numpy.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)
        

        final_inputs = numpy.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)
        

        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass 
    def save(self):	
    	numpy.savetxt('saved_wih.txt', self.wih)
    	numpy.savetxt('saved_who.txt', self.who)
    	pass
    def load(self):
    	self.wih = numpy.loadtxt('saved_wih.txt')
    	self.who = numpy.loadtxt('saved_who.txt')
    	pass
    # query the neural network 
    def query(self, inputs_list) : 
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

       
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
   
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
     
