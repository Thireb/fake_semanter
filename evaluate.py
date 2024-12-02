from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score
from icecream import ic
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    scorings = cross_val_score(
        model,
        x_train,
        y_train,
        cv=3,
        scoring="accuracy"
    )
    model_score = model.score(x_test, y_test)

    ic(scorings)
    ic(model_score)

    accuracy = accuracy_score(y_test, pred)
    ic(accuracy)

    confusion = confusion_matrix(y_test, pred)
    ic(confusion)


def confusion_matrix_heatmap(clf, x_test, y_test):
    confused = confusion_matrix(y_test, clf.predict(x_test),)
    confused_normal = np.zeros((confused.shape[0], confused.shape[1]))
    # ic(confused.shape(1))
    for column in range(confused.shape[1]):
        # print(column)
        confused_normal[:, column] = (confused[:, column] / sum(confused[:, column]))

    plt.ylim(-10, 10)
    sns.heatmap(
        confused_normal,
        cmap="Reds",
        annot=True,
        annot_kws={"size": 9}
    )
    locs, labels = plt.xticks()
    plt.xticks(locs, labels=("Pos", "Neu", "Neg"))
    locs, labels = plt.yticks()
    plt.yticks(locs, ("Pos", "Neu", "Neg"))
    plt.yticks(rotation=0)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual Sentiment Classification Percentage")


    bottom, top = plt.ylim()
    bottom += 0.5
    top -= 0.5
    plt.ylim(bottom, top)
    plt.show()






