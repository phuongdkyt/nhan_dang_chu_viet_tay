import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window
# helper to load data from PNG image files
import imageio
# glob helps select multiple files using patterns
import glob
#phần  khai báo lớp mạng neural
# neural network class definition
class neuralNetwork:
    
    
    # initialise the neural network
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes1, self.inodes))
        self.whh = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes2, self.hnodes1))
        self.who = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes, self.hnodes2))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T # 784 x node
        targets = numpy.array(targets_list, ndmin=2).T 
        
        # calculate signals into hidden layer
        hidden1_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden1_outputs = self.activation_function(hidden1_inputs)
        hidden2_inputs = numpy.dot(self.whh, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden2_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden2_errors = numpy.dot(self.who.T, output_errors)
        hidden1_errors = numpy.dot(self.whh.T, hidden2_errors)
        
        
        self.wih += self.lr * numpy.dot((hidden1_errors * hidden1_outputs * (1.0 - hidden1_outputs)), numpy.transpose(inputs))
        self.whh += self.lr * numpy.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)),numpy.transpose(hidden1_outputs))
        # update the weights for the links between the hidden2 and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden2_outputs))
         # update the weights for the links between the hidden1 and hidden2 layers
        
        # update the weights for the links between the input and hidden1 layers
        pass
        
    def save(self):
        numpy.save('saved_whh_2.npy', self.whh)
        numpy.save('saved_wih_2.npy', self.wih)
        numpy.save('saved_who_2.npy', self.who)
        pass
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden1_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden1_outputs = self.activation_function(hidden1_inputs)
        hidden2_inputs = numpy.dot(self.whh, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden2_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    def return_weights(self):
        wih = self.wih
        whh = self.whh
        who = self.who
        return wih, whh, who

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes1 = 200
hidden_nodes2 = 200
output_nodes = 26

# learning rate
learning_rate = 0.01

# create instance of neural networkn
n = neuralNetwork(input_nodes,hidden_nodes1, hidden_nodes2,output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("A_Z_HandwrittenData.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# train the neural network
# epochs is the number of times the training data set is used for training
epochs =10000
print("Start.......")
for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        n.save()
    print("\t\tHoc lan thu :{}".format(e+1))
    pass
print("End")

# our own image test data set
our_own_dataset = []

# load the png image data as test data set
for image_file_name in glob.glob('my_own_images2/2828_my_own_??.png'):
    
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])
    
    # load image data from png files into an array
    print ("loading ... ", image_file_name)
    img_array = imageio.imread(image_file_name, as_gray=True)
    
    # reshape from 28x28 to list of 784 values, invert values
    img_data  = 255.0 - img_array.reshape(784)
    
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))
    
    # append label and image data  to test data set
    record = numpy.append(label,img_data)
    our_own_dataset.append(record)
    pass

# test the neural network with our own images
arr = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O","P","Q","R","S","T","U","V","W","X","Y","Z"]
# record to test
item = 11

# plot image
matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

# correct answer is first value
correct_label = our_own_dataset[item][0]
# data is remaining values
inputs = our_own_dataset[item][1:]

# query the network
outputs = n.query(inputs) 
print (outputs)

# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)
for i, j in enumerate(arr):
    if(label == i):
        print("Du doan: ", j)
# append correct or incorrect to list
if (label == correct_label):
    print ("NHAN DANG THANH CONG!")
else:
    print ("SAI KHONG HOAN THANH NHAN DANG!")
    pass

wih, whh, who = n.return_weights()
fw_wih = numpy.savetxt("weights_wih.txt", wih)
fw_whh = numpy.savetxt("weights_whh.txt", whh)
fw_who = numpy.savetxt("weights_who.txt", who)
print("Write File Finish!!")
