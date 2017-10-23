import warnings
import pandas
import numpy 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter


#Returns a three-way split of the data
def train_validation_test_split(X, y, trainSize, validationSize, testSize):
    
    split1 = validationSize + testSize
    X_train, X_remainder, y_train, y_remainder = train_test_split(X, y, test_size=split1, random_state=0)
    X_validation, X_test, y_validation, y_test = train_test_split(X_remainder, y_remainder, test_size=0.5, random_state=0)
    return (X_train, X_validation, X_test, y_train, y_validation, y_test)

def predict(X_train, y_train, x_test, k): 
    
    dist = []
    neighbors = []
    
    for i in range(len(X_train)):
        distance = numpy.sqrt(numpy.sum(numpy.square(x_test - X_train[i, :])))
        dist.append([distance, i])
    dist = sorted(dist)
    for k in range(k):
        index = dist[k][1]
        neighbors.append(y_train[index])
    return Counter(neighbors).most_common(1)[0][0]


def neighbors(X_train, y_train, X_test, predictions, k):
    
    #Obtains the neighboring examples
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))
def main():
    
    df = pandas.read_csv('titanicdata.csv')
    
    #Drop examples with null values
    df = df.dropna(subset =['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
    #Convert attribute values to numeric type
    df.apply(pandas.to_numeric, errors='ignore')
    
    #Read in y values, convert to 0 or 1
    y = df.iloc[:, 1].values
    y = numpy.where(y == 0, 0, 1)
    
    #read in X values, convert sex to 0 or 1
    X = df.iloc[:, [2, 4, 5, 6, 7, 9]].values
    X[X == 'male'] = 0
    X[X == 'female'] = 1
    
    #Print the possible class labels
    print('Class labels:', numpy.unique(y))
    
    #Split the data into train, validation, and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = train_validation_test_split(X, y, 0.7, 0.15, 0.15)
    
    #Standardiize the data, ignoring type conversion warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_validation_std = sc.transform(X_validation)
        X_test_std = sc.transform(X_test)
    
    #Make predictions
    predictions = []
    neighbors(X_train_std, y_train, X_validation_std, predictions, 11)
    predictions = numpy.asarray(predictions)
    results = (y_validation == predictions)
    
    #Calculate performance metrics
    misclassified_examples = 0
    for r in results:
    	if r == False:
    		misclassified_examples = misclassified_examples + 1
    
    accuracy = 100* accuracy_score(y_validation, predictions)
    
    #Print out performance metrics
    print('Accuracy Score: %d' % accuracy + '%')
    print('Total misclassified: %d' % misclassified_examples)
    
main()


