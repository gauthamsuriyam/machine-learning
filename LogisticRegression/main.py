import pandas as pd
import numpy as np

np.set_printoptions(precision =3)


def data_transf(data_set):                      #performs feature scaling and one_hot encodings
  for i in data_set.columns:                    #for each feature country, occupation, etc
    recColDict = {}                             #dictionary for each feature and class (<'lable_Val',assigned_integer) 
    if(data_set[i].dtype == object):
      if(data_set[i].describe()['unique']==2 or i is data_set.columns[-1]):          #for columns with binary values, all strings are converted to 0 and 1 or if target column assign numerical values eg: 'nature_of_death' : 1, 'suicide': 1
        recList = data_set[i].values.tolist()
        recUnqList = set(recList)                #removing duplicates to create unique set of lists
        count=0
        for unq in recUnqList:                   #assign int_vals to each element
          if i is not data_set.columns[-1]:
            recColDict[unq] = count
          else:
            recColDict[unq] = count+1
            print("cause: "+unq+" value: "+str(count+1))
          count +=1
        #note: for testing/eval data only remap values to use same dictionary values
        data_set[i] = list(map(lambda x: recColDict[x],data_set[i]))          #remap all the original values to integer values
      else:
        if i is not data_set.columns[-1]:         #One_hot encoding for categorical data
          one_hot = pd.get_dummies(data_set[i])
          for ohCol in one_hot.columns:
            data_set.insert(data_set.columns.get_loc(i),i+ohCol,one_hot[ohCol])
          data_set = data_set.drop(i,axis=1)
    elif(data_set[i].dtype == np.int64 or data_set[i].dtype == np.float64): #For numerical values perform mean normalization
      colMean, colMin, colMax = data_set[i].describe()[['mean','min','max']]
      if i is not data_set.columns[-1]:         #mean normalization for feature scaling
        def mean_normalization(x):
          if((colMax - colMin) == 0):
            return 0
          elif(np.isnan(x)):
            #print("found nan val")
            return colMean                      #using mean value for missing values
          else:
            return ((x - colMean)/(colMax - colMin))
        data_set[i] = list(map(mean_normalization,data_set[i]))
  return data_set

def CostFunction(hypo,optVals):                 #Cost function - checks for convergence
  optVals = np.array(optVals)
  hypo = np.array(hypo)
  cost = optVals*(np.log(hypo)) + (1-optVals)*(np.log(1-hypo))
  return -(cost.sum()/len(cost))

def LogisticFn(z):                              #sigmoid function - hypothesis calculation
  return (1/(1+np.exp(-z)))

def UpdateWeights(hypo,weights, optVals,inpVals, learning_rate):  #perform Gradient decent to update weights
  #print(len(hypo))
  delta = np.dot(np.subtract(hypo,optVals),inpVals)
  weights = np.subtract(weights,delta*learning_rate/len(hypo))
  return weights

def LogisticRegression(data_set,weightSet,train = False,epoch = 1,learning_rate = 0.1,batchSize = 1,maxClassVal=1):
  #print("maxClassVal"+str(maxClassVal))
  #batchSize = 100                        #sub size of training set
  
  target = data_set.columns[-1]
  inpx = data_set.drop(target, axis =1)             #separating input values from target values
  x0 = np.ones(len(data_set))                       #initial input feature values
  inpx.insert(0,'x0',x0)
  opt = data_set[target]                            #Target variable reference  
  #print("test4: ")
  #naTest = inpx.isna()
  #cnt = 0
  #for v in range(len(naTest.columns)):
    #for indx in naTest[naTest.columns[v]]:
      #if indx is True:
        #print(cnt)
      #cnt +=1

  #for index in naTest.columns:
    #print(index)
    #print(True in naTest[index])
    #if(True in naTest[index]):
      #print("Error")
      #break
  #print(inpx.iloc[0])
  #print(inpx.iloc[0].values)
  #inpVals = inpx.values            #conv to list
  #print(inpVals)
  #print("test5")
  for e in range(epoch):
    
    hypoDf = pd.DataFrame()                         #data frame for ref of hypothesis values for each one_hot outputs
                                  
    #print(int(opt.describe()['max']))
    
    batchLen = int(len(inpx)/batchSize) #length of each batch
    print("max class val: "+str(maxClassVal))
    for b in range(0,int(len(inpx)),batchLen): #dividing entire set based on smaller batch size
      print(b)
      costSet = list()
      inpVals = inpx.iloc[b:batchLen+b].values
      
      hypoDfTemp = pd.DataFrame()
      #print(len(inpVals))
      for cls in range(maxClassVal):   #for multi class classification, performing training using one vs all
        print("cls: "+str(cls))
        optVals = opt.iloc[b:batchLen+b].values
        weights = weightSet[cls]
        def OneVall(x):
          if(x==cls+1):
            return 1
          else:
            return 0
        optVals = list(map(OneVall,optVals))    
        #print("optVals: "+str(optVals))
        z = np.dot(inpVals,weights)                   #dot product of weight and inputs
        hypo = list(map(LogisticFn,z))                #hypothesis values - applying sigmoid function
        
        if(train):                                    #Training mode
          #learning_rate = 10                          #learning_rate    
          weights = UpdateWeights(hypo,weights,optVals,inpVals,learning_rate)
          weightSet[cls] = weights

          costSet.append(CostFunction(hypo,optVals))
          #costVal = CostFunction(hypo,optVals)        #cost function
          #print("Cost: "+str(costVal))

        else:                                         #Evaluation mode
          #f cls+1 not in hypoDf.columns:
          hypoDfTemp[cls+1] = hypo                  
            #hypothesis for each class
          #else:
            #hypoDf[cls+1].append(np.array(hypo),ignore_index =False)
      #print(hypoDf)
      if not train:           #if in testing phase
        #print("hypoDfTemp: "+str(len(hypoDfTemp)))
        #print(hypoDf)
        #print(hypoDfTemp)
        if hypoDf.empty:
          hypoDf = hypoDfTemp
        else:
          #print("hypoDf not empty")
          hypoDf = hypoDf.append(hypoDfTemp,ignore_index = True)
          
        #print("hypoDf: "+str(len(hypoDf)))
      else:
        if(e==0 or (e+1)%100 == 0):
          print("epoch: "+str(e+1))
          print("Cost set: "+str(np.array(costSet)))
    if not hypoDf.empty:                            #display result set with % of correct classification
      print("size of hypo: "+str(len(hypoDf)))
      #print(hypoDf)
      hypoDf['hypo'] = hypoDf.idxmax(axis=1)
      hypoDf = hypoDf.drop([1,2,3,4],axis=1)
      hypoDf.insert(0,'cause',data_set[target].values)
      print(hypoDf)
      hypoDf['count'] = hypoDf['cause'] - hypoDf['hypo']
      ccVc = hypoDf['count'].value_counts(sort=False)

      if 0 not in ccVc:
        correctCount = 0
      else:
        correctCount = ccVc[0]
      #print(hypoDf['count'])
      print("correct class: "+str(correctCount))
      print("Correct classification percentage: "+str(100*correctCount/hypoDf['count'].count())+"%")  #percentage of correct classification 
    #else:
      #print(costSet)     
  return weightSet

def InitiateClassifier():
  #main module
  print("reading Input file")
  records = pd.read_csv('codDataSet2.csv')
  print("all records loaded")
  records = records.sample(frac=1).reset_index(drop=True)                    #randomizing the entire data set for training & test set
  records = records.drop(['human','date_of_birth','date_of_death'],axis=1)   #drop unsued data
  #records = records.drop(['education','balance','day','month','campaign','pdays'],axis=1)   #drop unsued data
  
  #records = records.drop(['Index'],axis=1)
  
  #print(records.head())
  target = records.columns[-1]
  
  if(records[target].dtype != np.int64):
    modSize = int(records[target].describe()['unique'])
  else:
    modSize = int(records[target].describe()['max'])
  #print(modSize)
  print("beginning data transformaiton..")
  records = data_transf(records)                                   #perfrom data transformation and scaling
  print("Transformation complete...")
  
  #separating training set and test set
  train_rec = records.iloc[:5000]     #training set
  test_rec = records.iloc[5000:6000]      #testing set
  #print(records.head())
  if(True):
    

    #begin training
    #print("col size: "+str(len(train_rec.columns)))
    weights = np.random.randn(modSize,len(train_rec.columns))       #weights include initial weight w0, and weights for rest of the features
    print("Before training")
    #print(weights)

    weights = LogisticRegression(test_rec,weights,maxClassVal = modSize)                  #evaluation set before training
  

    print("Beginning training..")
    weights = LogisticRegression(train_rec,weights,True,100,0.001,batchSize = 25,maxClassVal = modSize)        #initiate logistic regression to get    trained weights: training set, initial weights, if in training mode, epochs, learning_rate, batchsize = number of sets the input needs to be divided, maximum representation value of output
    #print("Training complete")
    #print("Beginning evaluation")
    weights = LogisticRegression(test_rec,weights,maxClassVal = modSize)                  #this is a set of weights(matrix) for multiclass classification
    #weights = LogisticRegression(train_rec,weights,True,200)
    #weights = LogisticRegression(test_rec,weights)
  return records

records = InitiateClassifier()                                    #Begin classification

#TODO: when all the outputs are wrong after classification, error is output since its not able to find 0
#TODO: levels are missing because the output set might not have al the values and the max can become smaller than in the original set