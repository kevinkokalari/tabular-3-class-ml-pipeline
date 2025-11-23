# Kevin Kokalari
# 2025-04-03

import pandas as pd
from pandas.api.types import is_integer_dtype
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def corrPlot(data, featuresToPlot):
    _, axes = plt.subplots(nrows=11, ncols=11, figsize=(20, 20))


    labels = data.y.unique()
    colors = ['blue', 'red', 'yellow']
    for ax_i, i in enumerate(featuresToPlot):
        for ax_j, j in enumerate(featuresToPlot):
            x = 'x' + str(i)
            y = 'x' + str(j)
            for k, label in enumerate(labels):
                data[data.y == label].plot(kind='scatter', x=x, y=y, ax=axes[ax_i][ax_j], 
                                      c=colors[k], )
                
            axes[ax_i][ax_j].set_xlabel(x)
            axes[ax_i][ax_j].set_ylabel(y)

def loadFromCSV(path):
        df = pd.read_csv(path, 
                 sep=',',         
                 encoding='utf-8'  
                )
        df = df.drop(columns=['Unnamed: 0'])
        try:
            print(df['y'].value_counts())
        except KeyError or InvalidIndexError:
            pass
        return df

def parseLoadedData(df):
        data = df

        #Convert bool to int
        data["x12"] = data["x12"].astype(int)

        x7_mapping = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Boom!": 5}
        y_mapping = {"Antrophic": 0, "OpenAI": 1, "Mistral": 2}

        data["x7"] = data["x7"].map(x7_mapping)

        try:
            data['y'] = data['y'].map(y_mapping)
        except KeyError or InvalidIndexError:
            pass
        #print(df.shape)
        #print(df.drop_duplicates().shape)
        #print(df.isnull().sum())
        #print(df['x7'].value_counts())
        #print(df.describe(include="all"))
        return data
    
def scaleFeatures(X_train, X_test):
        # Fit scaler on TRAIN only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

def boostFeatures(X_train, X_test):
    pca = PCA(n_components = 'mle', svd_solver = 'auto')

    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca  = pca.transform(X_test)

    return X_train_pca, X_test_pca


def saveResultLabels(y_pred, path='PredLabels.txt'):
        # Mappning från int → label
        y_mapping = {0: "Apple", 1: "Google", 2: "Meta"}
        #y_mapping = {0: "OpenAI", 1: "Antrophic", 2: "Mistral"}
        
        y_res = []

        for i in y_pred:
            y_res.append(y_mapping[i])

        # Om y_pred är en numpy-array eller vanlig lista av heltal
        with open(path, 'w', encoding='utf-8') as f:
            for label in y_res:
                f.write(f"{label}\n")
        return

def plotOutliers(data, threshold=4):
    numeric_df = data.select_dtypes(include=["float", "int"])

    z_score = np.abs(stats.zscore(numeric_df, nan_policy='omit').to_numpy())

    feature_names = numeric_df.columns.to_list()
    n_features = z_score.shape[1]

    ncols = int(np.ceil(n_features / 2))          # två rader, valfritt antal kolumner
    nrows = 2 if n_features > 1 else 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.ravel() if n_features > 1 else [axes]

    for i in range(n_features):
        ax = axes_flat[i]
        ax.plot(z_score[:, i], marker='o', linestyle='-', alpha=0.7)
        ax.axhline(threshold, color='red', linestyle='--', linewidth=1)
        ax.set_title(feature_names[i])
        ax.set_xlabel("Index (radnummer)")
        ax.set_ylabel("Z-score")

    # Ta bort överblivna ax-objekt
    for j in range(n_features, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.show()

        

   
       


if __name__ == "__main__":

    data = loadFromCSV("./Realting/Train.csv")
    parseLoadedData(data)
    #saveResultLabels(np.array([1,0,2,1,0]))