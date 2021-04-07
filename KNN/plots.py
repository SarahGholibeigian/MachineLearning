# import numpy, pandas, sklearn
from predict import *

def decisionboundaries(dataset, df, x_axis, y_axis, n_neighbors, weights):

        X = dataset.feature[[x_axis, y_axis]].to_numpy()
        y = dataset.label.to_numpy()

        # encoding the target column
        ord_enc = OrdinalEncoder()
        y_encoded = ord_enc.fit_transform(pd.DataFrame(dataset.label))

        # step size in the mesh
        h = .02

        # number of colors should be the same as the number of categories
        unique_target = len(dataset.label.unique())
        cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'][:unique_target])
        cmap_bold = ['darkorange', 'c', 'darkblue'][:unique_target]

        # we create an instance of Neighbours Classifier and fit the data.
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        clf.fit(X, y_encoded)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y,
                        palette=cmap_bold, alpha=1.0, edgecolor="black")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()
