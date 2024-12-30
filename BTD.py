from sklearn.tree import DecisionTreeClassifier
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

def cancerfeatures(Mean,Variance,Standard_Deviation,Entropy,Skewness,Kurtosis,Contrast,Energy,ASM,Homogeneity,Dissimilarity,Correlation,Coarseness):
    dataset = pd.read_csv('dataset = pd.read_csv('C:/Users/HP/Downloads/Brain Tumor Detection/Brain Tumor.csv')

    x=dataset.drop('Class',axis=1)
    y=dataset['Class']

    classifier=DecisionTreeClassifier()
	
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
    
    classifier.fit(x_train,y_train)

    predictedresult= classifier.predict(x_test)
  
    print("Result expected ", y_test)
    print("Result predicted by ML",predictedresult)
    
    Accuracy=metrics.accuracy_score(y_test, predictedresult)
    
    print((Accuracy)*100)
    
    tn,fp,fn,tp=confusion_matrix(y_test,predictedresult).ravel()
    
    print(tn,fp,fn,tp)
    
    
    
    Prediction=classifier.predict([[Mean,Variance,Standard_Deviation,Entropy,Skewness,Kurtosis,Contrast,Energy,ASM,Homogeneity,Dissimilarity,Correlation,Coarseness]])
    if Prediction==0:
        print("No Chances of Cancer")
    elif Prediction==1:
        print("Chances of Cancer")
   
    
def main():

    print("Enter the Mean")
    Mean=float(input())
    print("Enter the Variance")
    Variance=float(input())
    print("Enter the Standard Deviation")
    Standard_Deviation=float(input())
    print("Enter the Entropy")
    Entropy=float(input())
    print("Enter the Skewness")
    Skewness=float(input())
    print("Enter the Kurtosis")
    Kurtosis=float(input())
    print("Enter the Contrast")
    Contrast=float(input())
    print("Enter the Energy")
    Energy=float(input())
    print("Enter the ASM")
    ASM=float(input())
    print("Enter the Homogeneity")
    Homogeneity=float(input())
    print("Enter the Dissimilarity")
    Dissimilarity=float(input())
    print("Enter the Correlation")
    Correlation=float(input())
    print("Enter the Coarseness")
    Coarseness=float(input())
    
    cancerfeatures(Mean,Variance,Standard_Deviation,Entropy,Skewness,Kurtosis,Contrast,Energy,ASM,Homogeneity,Dissimilarity,Correlation,Coarseness)
if __name__=="__main__":
	main()


