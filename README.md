# Relational-Network-Pytorch-
This is a Pytorch implementation of paper [A Simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf) with interactive GUI Question-Answering Interface.

## Required Package
1. tqdm
2. tensorbardX
3. numpy
4. pytorch
5. matplotlib
6. colorlog
7. PyQt5
8. cv2
9. PIL
10. progressbar


### How to Run

#### Sort-of-Clevor Dataset 
Sort-of-CLEVR is simplified version of ![CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/) proposed by the authors.This is composed of 10000 images and 20 questions (8 relational questions and 12 non-relational questions) per each image. 6 colors (blue, green, red, yellow, magenta, cyan) are assigned to randomly chosen shape (square or circle), and placed in a image.

Non-relational questions are composed of 3 subtypes:

Shape of certain colored object
Horizontal location of certain colored object : whether it is on the left side of the image or right side of the image
Vertical location of certain colored object : whether it is on the upside of the image or downside of the image
Theses questions are "non-relational" because the agent only need to focus on certain object.

Relational questions are composed of 3 subtypes:

Color of the object which is closest to the certain colored object
Color of the object which is furthest to the certain colored object
These questions are "relational" because the agent has to consider the relations between objects.

Questions are encoded into a vector of size of 13 : 6 for one-hot vector for certain color among 6 colors, 2 for one-hot vector of relational/non-relational questions. 5 for one-hot vector of 5 questions.

#### Sort-of-Clevor Dataset Generation

go to directory "DataGenerator" and run the following command in terminal:

    python sort_of_clevr_generator.py
    
A folder "datasets/Sort-of-CLEVR_default" will be created which contains two files: data.hy and id.txt

data.hy contains images questions and answers while id.txt contains id corresponds to each triplet (image, question, answer).

#### Training RN Model 
go to the root directory and run the following command in terminal:
    
    python trainer.py
   
Note that training a RN module requires a GPU installed on your local machine. Based on my experience, the training time is approximately 30 minutes using a GTX 1060 (6GB) graphic card. The train:valid:test ratio is 75%:15%:15%. The overall test accuracy is approximately 95.933% 

#### Running an interactive QA program
go to the directory "InteractiveUI" and run the following command:

    python ui_main.py
  
Here is a demo showing how it works.
![](https://github.com/AlenUbuntu/Relational-Network-Pytorch-/blob/master/project/test.gif)
