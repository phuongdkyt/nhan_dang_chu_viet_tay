from tkinter import Tk, Frame, Scrollbar, Label, END, Entry, Text, VERTICAL, Button,Canvas,PhotoImage
import socket
from tkinter import filedialog

from PIL import ImageTk,Image  
from tkinter import messagebox
import numpy
import matplotlib
import glob
import imageio 
# library for plotting arrays
import matplotlib.pyplot as plt
import pylab

class GUI:

        def __init__(self, master):
            self.png=''
            self.input_nodes = 784
            self.hidden_nodes = 100
            self.output_nodes = 10
            self.learning_rate = 0.1
            #################
            self.iinput_nodes = 784
            self.ihidden_nodes1 = 200
            self.ihidden_nodes2 = 200
            self.ioutput_nodes = 26
            self.ilearning_rate = 0.01
            ###############33
            self.root = master
            self.frameResult = Frame( background="blue")
            self.label=None
            self.check=True
            self.join_button = None
            self.initialize_gui()
            
        def initialize_gui(self):
            self.root.title("Handwritten number recognition program")
            self.root.geometry("500x250+300+300")
            self.root.configure(background='#9999FF')
            self.root.resizable(0, 0)
            self.display_name_section()
            
            

         
        def display_name_section(self):
            frame = Frame()
            self.join_button = Button(frame, text="input file path", width=10, command=self.onOpen,bg='#FF6633').pack(side='left')
            frame.pack(side='top', anchor='nw')
            frame = Frame()
            
            self.join_button = Button(frame, text="reset", width=10, command=self.onDelete,padx=20,pady=5,font=("Serif", 12),bg='#FF6633').pack(side='bottom')
            Label(frame, text='reset                        ', font=("Serif", 12),bg='#9999FF').pack(side='bottom')
            
            self.join_button = Button(frame, text="start", width=10, command=self.onTrain2,padx=20,pady=5,font=("Serif", 12),bg='#FF6633').pack(side='bottom')
            Label(frame, text='word recognition    ', font=("Serif", 12),bg='#9999FF').pack(side='bottom')
            self.join_button = Button(frame, text="start", width=10, command=self.onTrain,padx=20,pady=5,font=("Serif", 12),bg='#FF6633').pack(side='bottom')
            Label(frame, text='number recognition', font=("Serif", 12),bg='#9999FF').pack(side='bottom')
            
            frame.pack(side='left')
        def onDelete(self):
            self.frameResult.destroy()
            self.label.destroy()
        def onOpen(self):
            self.frameResult.destroy()
            if(self.label!=None):    
                self.label.destroy()
            self.frameResult = Frame()
            filename =  filedialog.askopenfilename(title = "png",filetypes = (("PNG files","*.png"),("all files","*.*")))
            self.png=filename
            print(self.png)
        def onTrain(self):
            
            from network_claas import neuralNetwork
            n = neuralNetwork(self.input_nodes,self.hidden_nodes,self.output_nodes,self.learning_rate)
            # training_data_file = open("mnist_train_100.csv", 'r') 
            # training_data_list = training_data_file.readlines() 
            # training_data_file.close()

            # # # train the neural network 
            # # # epochs is the number of times the training data set is used for training 
            # epochs = 20
            
            # for e in range( epochs): 
            
            #     # go through all records in the training data set 
            
            #     for record in training_data_list: 
            #         # split the record by the ',' commas 
            #         all_values = record.split(',') 
            #         # scale and shift the inputs 
            #         inputs = (numpy.asfarray( all_values[1:]) / 255.0 * 0.99) + 0.01 
            #         # create the target output values (all 0.01, except the desired label which is 0.99) 
            #         targets = numpy.zeros(output_nodes) + 0.01 
            #         # all_values[0] is the target label for this record
            
            #         targets[int(all_values[0])] = 0.99 
            #         n.train(inputs, targets) 
            #     print("\t\tHoc lan thu :{}".format(e+1))
            #     pass 
            # n.save()
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
            performance=(scorecard_array.sum() / scorecard_array.size)*100
            t0='system performance:'+str(performance)+'%'
            our_own_dataset = []

            # load the png image data as test data set
            for image_file_name in glob.glob(self.png):
                
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
            item = 0

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
      
            
            t1="network says:"+str(label)+'              '
            t2='​accuracy:'+str(int(outputs[label]*100))+'% '
            Label(self.frameResult, text=t1, font=("Serif", 14)).pack(side='top', anchor='w')
            Label(self.frameResult, text=t2, font=("Serif", 14)).pack(side='top', anchor='w')
            Label(self.frameResult, text=t0, font=("Serif", 14)).pack(side='top', anchor='w')
            self.frameResult.pack(side='top')
            image = Image.open(self.png)
            image = image.resize((100, 100), Image.ANTIALIAS) 
            photo = ImageTk.PhotoImage(image)
            self.label = Label(image=photo)
            self.label.image = photo # keep a reference!
            self.label.pack()
            
        def onTrain2(self):
            from network import neuralNetwork
            n = neuralNetwork(self.iinput_nodes,self.ihidden_nodes1, self.ihidden_nodes2,self.ioutput_nodes, self.ilearning_rate)
            n.load()

            our_own_dataset = []

            # load the png image data as test data set
            for image_file_name in glob.glob(self.png):
                
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
            item = 0

            # plot image
            matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

            # correct answer is first value
            correct_label = our_own_dataset[item][0]
            # data is remaining values
            inputs = our_own_dataset[item][1:]

            # query the network
            outputs = n.query(inputs) 
            print (outputs)

            # the index of the highest value corresponds to the 
            kitu=''
            label = numpy.argmax(outputs)
            for i, j in enumerate(arr):
                if(label == i):
                    print("Dự đoán: ", j)
                    kitu=j
                    break;
                
            t1="network says "+j+'             '
            t2='​accuracy:'+str(int(outputs[label]*100))+'% '
            Label(self.frameResult, text=t1, font=("Serif", 14)).pack(side='top', anchor='w')
            Label(self.frameResult, text=t2, font=("Serif", 14)).pack(side='top', anchor='w')
            self.frameResult.pack(side='top')
            image = Image.open(self.png)
            image = image.resize((100, 100), Image.ANTIALIAS) 
            photo = ImageTk.PhotoImage(image)
            self.label = Label(image=photo)
            self.label.image = photo # keep a reference!
            self.label.pack()
            # append correct or incorrect to list
            # if (label == correct_label):
            #     print ("NHAN DANG THANH CONG!")
            # else:
            #     print ("SAI KHONG HOAN THANH NHAN DANG!")
            #     pass
if __name__ == '__main__':
    root = Tk()
    gui = GUI(root)
    root.mainloop()
    
