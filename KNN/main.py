# import numpy, pandas, sklearn
from imports import

# import defined classes, attributes and methods
from predict import *

# import prints function
import prints

# import decisionboundaries function
import plots

# read .csv file
df = pd.read_csv("/home/mgh/CS-Tutorial/Extra/KNN/dataset/df_final_normalized.csv")

# define label and index columns
label = 'regime'
index_label = 'Unnamed: 0'

# create the dataset object
test_size = 0.20
Data = Dataset(dataframe=df, label=label, test_size=test_size)

# create knn model and perform predictions
n_neighbors = 15
weights = 'uniform' # 'distance'

knn = KNN(dataset=Data, n_neighbors=n_neighbors, weights=weights)

# call the print commands
prints.confusion_matrix(model=knn, n_neighbors=n_neighbors , weights=weights)
prints.classification_report(model=knn, n_neighbors=n_neighbors , weights=weights)
prints.accuracy_score(model=knn, n_neighbors=n_neighbors , weights=weights)
prints.precision_score(model=knn, n_neighbors=n_neighbors , weights=weights)
prints.F1_score_test(model=knn, n_neighbors=n_neighbors , weights=weights)

# plot decision_boundaries
feature_1 = 'water_T'
feature_2 = 'O2Sat'
plots.decisionboundaries(dataset=Data, df=df, x_axis=feature_1, y_axis=feature_2, n_neighbors=n_neighbors, weights=weights)
