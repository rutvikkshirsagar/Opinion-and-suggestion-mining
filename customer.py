import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
     FigureCanvasTkAgg)
import tkinter as tk
import numpy as np
import nltk
from nltk.corpus import stopwords
import pandas as pd
from nltk import FreqDist
import seaborn as sns
nltk.download('stopwords')
import openai
import joblib
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox


model = joblib.load(Path('model_joblib'))

#seggregating happy unhappy reviews customer perspective

def happy_unhappy(cols_as_np):
    #All review predictions
    arr=[]     
    #negetive reviews added
    bad_review_array_index=[]
    bad_review_array=[]             

    for i in range(0,len(cols_as_np)):
    
        exp=[cols_as_np[i]]
        results=model.predict(exp)
        arr.append(results[0])
        if(results[0]=='not happy'):
            bad_review_array_index.append(i)
            bad_review_array.append(cols_as_np[i])
            
            
            
    h=arr.count('happy')
    nh=arr.count('not happy')
    h_percent=(h*100)/(h+nh)
    nh_percent=(nh*100)/(h+nh)

   
    a=[round(h_percent),round(nh_percent)]
    return a
    
    



def input_file_customer(a):
    Reviewdata=[]
    

    
    Reviewdata=a
    
    
    

    cols_as_np = Reviewdata['Reviews'].to_numpy()
   
        


    prediction=happy_unhappy(cols_as_np)
    return prediction
    











'''
# Define a function to preprocess the reviews and train the classifier
def train_classifier():
    # Get the path of the csv file
    csv_path = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("CSV files","*.csv"),("all files","*.*")))
    if not csv_path:
        return

    # Load the csv file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Preprocess the reviews by converting them to lowercase and removing punctuations and stopwords
    vectorizer = CountVectorizer(lowercase=True, stop_words='english', strip_accents='ascii', token_pattern=r'\b[a-zA-Z]{3,}\b')
    X = vectorizer.fit_transform(df['Review'])

    # Train a Naive Bayes classifier on the reviews and their sentiment labels
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X, df['Sentiment'])

    # Predict the sentiment labels for the reviews and store them in a new column in the DataFrame
    df['Predicted_Sentiment'] = nb_classifier.predict(X)

    # Create a confusion matrix to show the performance of the classifier
    confusion = confusion_matrix(df['Sentiment'], df['Predicted_Sentiment'])

    # Calculate the accuracy of the classifier
    accuracy = ((confusion[0][0] + confusion[1][1]) / sum(sum(confusion))*100)

    # Show the accuracy of the classifier in a message box
    messagebox.showinfo("Accuracy", f"Accuracy: {accuracy:.2f}%")

    # Plot a bar chart to show the number of positive and negative reviews
    positive_count = df[df['Sentiment'] == 'happy'].shape[0]
    negative_count = df[df['Sentiment'] == 'not happy'].shape[0]
    plt.bar(['Positive', 'Negative'], [positive_count, negative_count])
    plt.title("Positive vs Negative Reviews")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

# Create the main window
root = Tk()
root.title("Sentiment Analysis")

# Create a button to train the classifier
train_button = Button(root, text="Train Classifier", command=train_classifier)
train_button.pack()

root.mainloop()











# from tkinter import filedialog
# from tkinter import *

# root = Tk()

# def browse_files():
#     file_path = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("CSV files","*.csv"),("all files","*.*")))
#     print(file_path) # or do something else with the file_path

# browse_button = Button(root, text="Browse Files", command=browse_files)
# browse_button.pack()

# root.mainloop()




# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import confusion_matrix

# # Accept csv file path from user
# csv_path = input("Please enter the path of the csv file: ")

# # Load the csv file into a pandas DataFrame
# df = pd.read_csv(csv_path)

# # Preprocess the reviews by converting them to lowercase and removing punctuations and stopwords
# vectorizer = CountVectorizer(lowercase=True, stop_words='english', strip_accents='ascii', token_pattern=r'\b[a-zA-Z]{3,}\b')
# X = vectorizer.fit_transform(df['Review'])

# # Train a Naive Bayes classifier on the reviews and their sentiment labels
# nb_classifier = MultinomialNB()
# nb_classifier.fit(X, df['Sentiment'])

# # Predict the sentiment labels for the reviews and store them in a new column in the DataFrame
# df['Predicted_Sentiment'] = nb_classifier.predict(X)

# # Create a confusion matrix to show the performance of the classifier
# confusion = confusion_matrix(df['Sentiment'], df['Predicted_Sentiment'])
# # print("Confusion Matrix:")
# # print(confusion)

# # Calculate the accuracy of the classifier
# accuracy = ((confusion[0][0] + confusion[1][1]) / sum(sum(confusion))*100)
# print(f"Accuracy: {accuracy:.2f}%")

# # Plot a bar chart to show the number of positive and negative reviews
# positive_count = df[df['Sentiment'] == 'happy'].shape[0]
# negative_count = df[df['Sentiment'] == 'not happy'].shape[0]
# plt.bar(['Positive', 'Negative'], [positive_count, negative_count])
# plt.title("Positive vs Negative Reviews")
# plt.xlabel("Sentiment")
# plt.ylabel("Count")
# plt.show()

'''





import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox


def plot(h,uh):
    plt.bar(['Positive', 'Negative'], [h, uh])
    plt.title("Positive vs Negative Reviews")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()



def train_classifier():
    # Get the path of the csv file
    csv_path = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("CSV files","*.csv"),("all files","*.*")))
    if not csv_path:
        return

    # Load the csv file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    ans= input_file_customer(df)
    #a=plot(ans[0],plot[1])
    # Preprocess the reviews by converting them to lowercase and removing punctuations and stopwords
    

    


    label3 = Label( root, text = "Here is what other customers think about the product",
               bg = "white",font=('Times', 20))
  
    label3.pack(pady = 5)

    plt.bar(['Positive', 'Negative'], [ans[0], ans[1]])
    plt.title("Positive vs Negative Reviews")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    



    #label2 = Label( root, text = f"{a}",
          #     bg = "white",font=('Times', 20),wraplength=1500)
  
   # label2.pack(pady = 50)

   

   

   
    # Show the accuracy of the classifier in a message box
    #messagebox.showinfo("Suggestions", f"{ans}")

    
# Create the main window
root = Tk()

width = root.winfo_screenwidth()
height = root.winfo_screenheight()
root.geometry(f"{width}x{height}")
root.configure(bg='white')


#root.geometry("1400x1400+0+0")
root.title("Sentiment Analysis")
#train_classifier()
# Create a button to train the classifier

label1 = Label( root, text = "Get to know about the product",
               bg = "white",font=('Times', 24))
  
label1.pack(pady = 50)

train_button = Button(root, text="Upload Review File",height= 2, width=20,font=('Times', 15), pady=5,command=train_classifier)
train_button.pack()


root.mainloop()
