import warnings
import pandas as pd
from pandas import plotting as plt
import matplotlib
import matplotlib.pyplot as plot
import numpy as np
from sklearn import preprocessing as pp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics

warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')


def null_crime(file):

    # create a new data frame from the file, lets call it analysis

    analysis = pd.read_csv(file,
                           header=0,          # header in the first row
                           na_values='',      # na values are '' in csv
                           index_col=None)    # don't set an index column

    # show the data structure of the file
    print("Original:", analysis.head())

    # drop division and region columns (who needs that?)
    analysis.drop(analysis.columns[0:2], axis=1, inplace=True)

    print("Original, refined:", analysis.head())

    # draw a matrix of scatter plots
    plt.scatter_matrix(analysis)

    plot.figure('original')
    plot.hist(analysis['year'])
    plot.show()

    ####################################################################################################################
    # !feature scaling!
    # create a scale object
    scale = pp.MinMaxScaler()

    # pick columns to be scaled
    columns = ['year', 'crime_rate']

    # fit the scale object to the columns & set values
    analysis[columns] = scale.fit_transform(analysis[columns])
    print('\nScaled:\n', analysis.head())

    # Distribution of year After Scaling
    plot.figure('scaled')

    plot.hist(analysis['year'])
    plot.show()

    # use the inverse transformation to get and set original values
    analysis[columns] = scale.inverse_transform(analysis[columns])
    print('\nReturned to Original:\n', analysis.head())

    # create dummy vars for region, division and state columns

    '''You can get k-1 dummy levels by specifying drop_first = True, but we want to include all of them so we can pivot back 
    correctly using idxmax. This means we need to remove the last dummy when we generate our x inputs (see below) '''

    # create dummies on state column and don't add any prefix or sep (to get good state names when we pivot back)
    analysis = pd.get_dummies(analysis, columns=['State'], prefix='', prefix_sep='')
    print('\nDummy Variable Encoding:\n', analysis.head())

    ####################################################################################################################
    # !Evaluate the Accuracy of Linear Regression!
    # Set seed for reproducible work
    ''' Sklearn uses the numpy seed so we need to set the seed using numpy (np)'''
    np.random.seed(100)

    # Define X and Y variables for regression
    # only keep values where dependent variable is not null (otherwise will mess regression)
    no_null = analysis.dropna(how='any')

    # y = dependent = divorce rate
    y = no_null['crime_rate']

    # x = all other variables except for last dummy variable (so we have k-1): could specify by location as well
    x = no_null.drop(['crime_rate', 'Wyoming'], axis=1)

    # Hold out Method for evaluating accuracy metrics
    ''' You want to split the data into a training and testing set. You may also see "validation set', but
    you do not need to worry about evaluation for this assignment. 
    Training: the data used to build the model
    Validation: the data used to tune parameters
    Testing: the data used to get accuracy/performance measures for data outside of our initial sample'''

    # split the data into 80% train and 20% test ################
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # build regression model on training data ###################
    lm_test = LinearRegression().fit(x_train, y_train)

    # evaluate the model on the testing data to get R2 ##########
    ''' R2 = percent of variance of y explained by model '''
    r2 = lm_test.score(x_test, y_test)
    print('\nHold Out R2:', round(r2, 4))

    # get predictions and compare with actual to get RMSE #######
    # Make predictions using the testing set
    pred = lm_test.predict(x_test)

    # Get RMSE from predictions & actual values (y) from testing set
    ''' RMSE: average difference of predicted vs. actual in Y units'''
    # RMSE is the square root of MSE
    rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
    print('Hold Out RMSE:', round(rmse, 4))

    ####################################################################################################################
    # !K-Fold Cross Validation!
    ''' 'Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
    It generally results in a less biased or less optimistic estimate of the model skill than other methods, 
    such as a simple train/test split.' This can be used for validation (and parameter tuning), but also for getting
    less biased accuracy/performance metrics for the model (metrics will depend less on how data is split). 
    From: https://machinelearningmastery.com/k-fold-cross-validation/'''

    # create an empty linear regression object
    lm_k = LinearRegression()

    # get the R2 and MSE for each model generated with k-fold (number of models = k)
    # the default for cross_val score is R2, take 10 folds b/c common in literature
    scores_r2 = list(np.round(cross_val_score(lm_k, x, y, cv=10), 4))
    # set scoring parameter to get neg mse (use 10 folds for everything to keep consistency)
    scores_mse = cross_val_score(lm_k, x, y, cv=10, scoring="neg_mean_squared_error")
    # to get rmse we need to take the square root of the absolute mse values
    scores_rmse = list(np.round(np.sqrt(np.abs(scores_mse)), 4))
    print('\nCross-validated R2 By Fold:\n', scores_r2, "\nCross-Validated MSE By Fold:\n", scores_rmse)

    ####################################################################################################################
    # Get the overall R2 and MSE for this type of model on the data
    # generate a prediction list
    predictions = cross_val_predict(lm_k, x, y, cv=10)

    # compare prediction vs actual values to get metrics
    # r2
    r2 = metrics.r2_score(y, predictions)
    # take square root of mse to get rmse
    rmse = np.sqrt(metrics.mean_squared_error(y, predictions))
    print('Cross-Validated R2 For All Folds:', round(r2, 4), '\nCross-Validation RMSE For All Folds:', round(rmse, 4))

    ####################################################################################################################
    # !Build Final Model!

    # We want to build the final model with all the data
    lm_final = LinearRegression().fit(x, y)

    # Prep Data for Output
    # Predict With Model to Fill Null Values

    # get the row number (index #) of null values in crime rate
    crime_null_list = analysis.index[analysis['crime_rate'].isnull()].tolist()

    # get the x values for use in the prediction of null divorce rates (everything but divorce rate & dropped dummy)
    pred_vals = analysis.drop(['crime_rate', 'Wyoming'], axis=1)

    # for each row where crime rate is null, set crime rate to be the prediction generated by model
    # analysis and pred_values will have the same index, so can access the corresponding data from row position
    for x in crime_null_list:  # x is the position of the null value row
        # go to row x and col div_rate in analysis, replace value with predictions generated from corresponding x vals
        analysis.ix[x, 'crime_rate'] = lm_final.predict([pred_vals.iloc[x].tolist()])

    ####################################################################################################################
    # !Reshape to Remove Dummies!
    # get only the dummy columns and set them to a new dataframe (row location is preserved)
    state = analysis.drop(analysis.columns[:2], axis=1)

    # remove all dummy columns from analysis dataframe
    analysis.drop(analysis.columns[2:], axis=1, inplace=True)

    # create new variable state that is the column name of the highest value (1) for each row of dummies
    analysis['State'] = state.idxmax(axis=1)  # axis = 1 tells it to get the column name
    print('\nFinal Dataset:\n', analysis.head())

    analysis.to_csv("hwk5_crime.csv", sep=',', index=False)


null_crime("clean_crime.csv")
