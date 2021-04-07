def confusion_matrix(model, n_neighbors, weights):
    print(f"Confusion_matrix_train for k={n_neighbors}, weights={weights}:", model.train_confusion_matrix)
    print(f"Confusion_matrix_test for k={n_neighbors}, weights={weights}:", model.test_confusion_matrix)
    #plot_confusion_matrix(model.knn_model, Data.feature_test, Data.label_test)

def classification_report(model, n_neighbors, weights):
    print(f"Classification_report_train for k={n_neighbors}, weights={weights}:", model.train_classification_report)
    print(f"Classification_report_test for k={n_neighbors}, weights={weights}:", model.test_classification_report)

def accuracy_score(model, n_neighbors, weights):
    print(f"Accuracy_score_train for k={n_neighbors}, weights={weights}:", model.train_accuracy_score)
    print(f"Accuracy_score_test for k={n_neighbors}, weights={weights}:", model.test_accuracy_score)

def precision_score(model, n_neighbors, weights):
    print(f"Precision_score_train for k={n_neighbors}, weights={weights}:" , model.train_precision_score)
    print(f"Precision_score_test for k={n_neighbors}, weights={weights}:" , model.test_precision_score)

def F1_score_test(model, n_neighbors, weights):
    print(f"F1_score_train for k={n_neighbors}, weights={weights}:" , model.train_f1_score)
    print(f"F1_score_test for k={n_neighbors}, weights={weights}:" , model.test_f1_score)