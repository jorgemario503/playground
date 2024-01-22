import pandas as pd
import numpy as np
from scipy import stats # normality test
import pingouin as pg # correlation test
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# import modelling libraries
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# import model selector libraries
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from lazypredict.Supervised import LazyClassifier

# import visualization libraries
import matplotlib.pyplot as plt

# import neural network libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf


# SELECT FINAL FEATURES USING CORRELATION, NORMALITY TEST AND FEATURE IMPORTANCE
def eda_approval(target, dataset, ingresos):
    if target in dataset.columns:
        df = dataset.copy()
        df.insert(loc=0, column=target, value=np.where(df[target]==0,0,1))
        print(target,'default rate:',df[target].sum()/len(df))

        # Correlación y normalidad de residuos
        print('\nCORRELATION AND NORMALITY TEST')
        exclude_list = []
        for column in df.columns:
            # Cálculo de correlación, significancia e intervalos con pingouin
            pearson_corr = pg.corr(df[target], df[column], method='pearson')
            spearman_corr = pg.corr(df[target], df[column], method='spearman')
            kendall_corr = pg.corr(df[target], df[column], method='kendall')
            if column != target and abs(pearson_corr['r'][0]) > 0.3:
                print(f"{column} correlation with target: pearson={pearson_corr['r'][0].round(3)} with p-value={pearson_corr['p-val'][0].round(5)}, spearman={spearman_corr['r'][0].round(3)} with p-value={spearman_corr['p-val'][0].round(5)}, kendall={kendall_corr['r'][0].round(3)} with p-value={kendall_corr['p-val'][0].round(5)}")
            if pearson_corr['r'].isnull().values.any():
                print(f"{column} has a null correlation")
                exclude_list.append(column)
            
            # Cálculo de correlación parcial con salario
            if column != target and column != ingresos:
                partial_corr = pg.partial_corr(data=df, x=column, y=target, covar=ingresos)
                if abs(partial_corr['r'][0]) > 0.3:
                    print(f"{column} partial correlation={partial_corr['r'][0].round(3)} with p-value={partial_corr['p-val'][0].round(5)}")

            # Cálculo de normalidad con Shapiro-Wilk y D'Agostino
            statistic, sha_pvalue = stats.shapiro(df[column])
            k2, k_pvalue = stats.normaltest(df[column])
            if k_pvalue>0.05 or sha_pvalue>0.05:
                print(f"{column} normality tests: shapiro p-value={sha_pvalue}, d'agostino p-value={k_pvalue}")

        # Exclude variables with null correlation and equal correlation
        X = df[df.columns.difference(exclude_list)]
        # Separate dataframe into input and output
        y = X[target]
        X = X[X.columns.difference([target])]
        print(X.shape)

        # Feature importance by model
        print('\nFEATURE IMPORTANCE BY MULTIPLE MODELS')
        models = [DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier]
        exclude_list2 = []
        for modelling in models:
            model = modelling()
            model.fit(X, y)
            # Creating importances dataframe
            importances = pd.DataFrame({"feature_names" : model.feature_names_in_, 
                                    "importances" : model.feature_importances_}).sort_values(by='importances', ascending=False)
            # Exclude variables with 0 importance
            for i in range(len(importances)):
                if (importances["importances"][i] == 0) & ('giro' in importances["feature_names"][i]):
                    exclude_list2.append((modelling.__name__,importances["feature_names"][i]))

        # Exclude the repeated variables with 0 importance
        exclude = [exclude_list2[i][1] for i in range(len(exclude_list2))]
        from collections import Counter
        exclude_list2_final = [text for text, freq in Counter(exclude).items() if freq > 1]
        print("List of repeated variables with 0 importance: ",exclude_list2_final)

        # Final dataset
        X = X[X.columns.difference(exclude_list2_final)]

        # Split dataset in training and test
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state=123)

        print("\nDataset records and variables:", X.shape, "\n", X.columns.to_list())
        print("Train dataset:", X_train.shape, "\n", "Test dataset:", X_test.shape)
        
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

    else:
        print('no modelling yet')  


# RUN DIFFERENT COMPONENTS TO OPTIMIZE PCA 
def pca_optimization(X_train,y_train,X_test,y_test, max_component):
    print('\nPCA OPTIMIZATION')
    # Check number of PCA components
    components = [i for i in range(1,max_component+1) if i%5 == 0]
    models = [LogisticRegression(random_state=123)
                , GaussianNB()
                , PassiveAggressiveClassifier(C=0.05, loss='hinge', max_iter= 100, tol=0.1, random_state=5)
                , NearestCentroid()
                , ExtraTreesClassifier(random_state=123)
                ]

    for i in range(len(components)):
        print("\nNumber of components:",components[i])
        pca = PCA(n_components=components[i])
        train_pca = pca.fit_transform(X_train)
        print(pca.explained_variance_ratio_.sum())
        test_pca = pca.transform(X_test)

        # Check different models
        for model in models:
            # Fit different models using PCA components
            clf = model.fit(train_pca, y_train)
            # Predictions
            clf_pred = clf.predict(test_pca)
            # Precision on class 0 (ever PAR30 null)
            print(f"Precision {model}:",precision_score(y_test, clf_pred, pos_label=0))


# RUN PCA FOR DIFFERENT MODELS
def pca_approval(X_train,y_train,X_test,y_test, n_components):
    print('\nPCA COMPONENTS')
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(X_train)
    test_pca = pca.transform(X_test)

    # Pick the models
    log_pca = LogisticRegression(random_state=123)
    gnb_pca = GaussianNB()
    nc_pca = NearestCentroid()
    knn_pca = KNeighborsClassifier()
    extra_pca = ExtraTreesClassifier(random_state=123)
    bag_pca = BaggingClassifier(random_state=123)
    rfc_pca = RandomForestClassifier(random_state=123)
    ada_pca = AdaBoostClassifier(random_state=123)
    lgbm_pca = LGBMClassifier(random_state=123)
    lin_pca = LinearSVC(random_state=123)
    pag_pca = PassiveAggressiveClassifier(C=0.05, loss='hinge', max_iter= 100, tol=0.1, random_state=5)

    models = [log_pca, gnb_pca, nc_pca, knn_pca, extra_pca, bag_pca, rfc_pca, ada_pca, lgbm_pca, lin_pca, pag_pca]

    # Fit and predict different models using PCA components
    for model in models:
        # Fit
        model.fit(train_pca, y_train)
        # Predict
        pred = model.predict(test_pca)
        # Print results
        print('Confusion matrix:',model,'\n',confusion_matrix(y_test, pred),accuracy_score(y_test, pred),"\nPrecision:",precision_score(y_test, pred, pos_label=0),'\n')
    
    return {'X_train_pca':train_pca,'X_test_pca':test_pca
            ,'log':log_pca,'gnb':gnb_pca,'nc':nc_pca,'knn':knn_pca
            ,'extra':extra_pca,'bag':bag_pca,'rfc':rfc_pca,'ada':ada_pca,'lgbm':lgbm_pca
            ,'lin':lin_pca,'pag':pag_pca
            }


# RUN LAZYPREDICT WITH PCA
def pca_lazy(X_train,y_train,X_test,y_test, n_components):
    # Apply PCA to reduce dimensionality
    n_components = n_components
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(X_train)
    print(pca.explained_variance_ratio_.sum())
    test_pca = pca.transform(X_test)

    # Plotting PCA explained variance
    plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.title('PCA explained variance')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    # LazyPredict classification models with PCA components
    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    models_clf,predictions_clf = clf.fit(train_pca, test_pca, y_train, y_test)
    models_clf.sort_values('F1 Score', ascending=False)


# NEURAL NETWORK
def nn_approval(X_train,y_train,X_test,y_test):
    print('\nNEURAL NETWORK')
    # Normalization as array
    sc = StandardScaler()
    Xn_train = sc.fit_transform(X_train)
    Xn_test = sc.fit_transform(X_test)
    # Define and compile the model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=Xn_train.shape[1]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['BinaryAccuracy'])
    # Train the model (1)
    model.fit(Xn_train, y_train, epochs=10, validation_data=(Xn_test, y_test), verbose=0)
    
    # Add input layer and hidden layers
    model.add(Dense(units=64, activation='relu', input_dim=Xn_train.shape[1]))
    model.add(Dense(units=16, activation='relu'))
    # Add output layer with sigmoid activation for binary classification
    model.add(Dense(units=1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['BinaryAccuracy'])
    # Train the model (2)
    model.fit(Xn_train, y_train, epochs=100, batch_size=100, verbose=0)

    # Make predictions on the test set
    y_pred = model.predict(Xn_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    # Calculate accuracy
    precision = precision_score(y_test, y_pred_binary, pos_label=0)
    print("Precision:", precision)


# RECURRENT NEURAL NETWORK
def rnn_approval(X_train,y_train,X_test,y_test):
    print('\nRECURRENT NEURAL NETWORK')
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoader for the training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Define the RNN model
    class LoanApprovalRNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(LoanApprovalRNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_layers, self.hidden_size)
            out, hn = self.rnn(x, h0)
            out = self.fc(out[:,:])
            out = torch.sigmoid(out)
            return out

    # Initialize the model and set the device (CPU or GPU)
    X = pd.concat([X_train,X_test])
    model = LoanApprovalRNN(input_size=X.shape[1], hidden_size=128, num_layers=2)
    device = torch.device("cpu")
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs.view(-1), batch_y.float())
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    # Evaluation on test data
    model.eval()
    with torch.no_grad():
        test_X, test_y = X_test_tensor.to(device), y_test_tensor.to(device)
        test_outputs = model(test_X)
        predicted_labels = (test_outputs >= 0.5).view(-1,1)
        accuracy = (predicted_labels == test_y).float().mean()

    print(f"Test Accuracy: {accuracy.item():.2f}")


# CONVOLUTIONAL NEURAL NETWORK
def cnn_approval(X_train,y_train,X_test,y_test):
    print('\nCONVOLUTIONAL NEURAL NETWORK')
    # Normalization as array
    sc = StandardScaler()
    Xn_train = sc.fit_transform(X_train)
    Xn_test = sc.fit_transform(X_test)
    # Reshape the data for CNN input
    X_train_reshaped = Xn_train.reshape(-1, X_train.shape[1], 1)
    X_test_reshaped = Xn_test.reshape(-1, X_test.shape[1], 1)

    # Define the CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['BinaryAccuracy'])

    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=100, batch_size=100, verbose=0)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test_reshaped)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate precision
    precision = precision_score(y_test, y_pred_binary, pos_label=0)
    print("Precision:", precision)
