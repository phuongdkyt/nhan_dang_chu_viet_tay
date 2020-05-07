

from network_claas import neuralNetwork
import numpy
import matplotlib
import glob
import imageio 
# library for plotting arrays
import matplotlib.pyplot as plt
import pylab


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1



n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
n.load()
 # load the mnist test data CSV file into a list 
test_data_file = open("mnist_test_10.csv", 'r') 
test_data_list = test_data_file.readlines() 
test_data_file.close() 

# test the neural network 
# scorecard for how well the network performs, initially empty 
scorecard = [] 

# go through all the records in the test data set 
for record in test_data_list: 
    # split the record by the ',' commas 
    all_values = record.split(',') 
    # correct answer is first value 
    correct_label = int( all_values[ 0]) 
    # scale and shift the inputs 
    inputs = (numpy.asfarray( all_values[ 1:]) / 255.0 * 0.99) + 0.01 
    # query the network 
    outputs = n.query( inputs) 
    # the index of the highest value corresponds to the label 
    label = numpy.argmax( outputs) 
    # append correct or incorrect to list 
    if (label == correct_label): 
        # network' s answer matches correct answer, add 1 to scorecard 
        scorecard.append( 1) 
    else: 
        # network' s answer doesn' t match correct answer, add 0 to scorecard 
        scorecard.append( 0) 
        pass 
    pass 

# calculate the performance score, the fraction of correct answers 
scorecard_array = numpy.asarray( scorecard) 
print ("performance = ", scorecard_array.sum() / scorecard_array.size)
our_own_dataset = []

# load the png image data as test data set
for image_file_name in glob.glob('my_own_images/2828_my_own_?.png'):
    
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

# record to test
item = 2

# plot image
plt.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

# correct answer is first value
correct_label = our_own_dataset[item][0]

# data is remaining values
inputs = our_own_dataset[item][1:]

# query the network
outputs = n.query(inputs)
print (outputs)

# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)
print("network says ", label)
# append correct or incorrect to list
if (label == correct_label):
    print ("match!")
else:
    print ("no match!")
    pass

pylab.show() 

