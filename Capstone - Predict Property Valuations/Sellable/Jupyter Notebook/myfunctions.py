import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn import metrics
from sklearn.linear_model import ElasticNet, ElasticNetCV


def file_details(filename):
    print ('\n','='*10, 'DATA FIELD TYPES', '='*10, '\n', filename.dtypes)         # Display type of fields
    print ('\n','='*10, 'FILE SHAPE',       '='*10, '\n', filename.shape, '\n')    # Display how many rows and columnsS
    print ('\n','='*10, 'FILE INFORMATION', '='*10, '\n', filename.info())         # Display the info of each column & file
    print ('\n','='*10, 'STATISTICS',       '='*10, '\n', filename.describe(include='all')) # display file statistics
    print ('\n','='*10, 'ISNULL',           '='*10, '\n', filename.isnull().sum()) # Find all Nulls in file
    print ('\n','='*10, 'COLUMNS',          '='*10, '\n', filename.columns)        # Column Names
    print ('\n','='*10, 'HEAD()',           '='*10, '\n', filename.head())         # View the first 10 records
    print ('\n','='*10, 'TAIL()',           '='*10, '\n', filename.tail())         # Check last 5 records in the file
    filename.sample(5)
    return
    

def unique_type(filename, dtype):
    list = ['all','int64','float','object']                     # Possible list entries
    dtype = dtype[:3].lower()                                   # Store Arg type as 3 byte lowercase
    for string in list:                                         # Read thru the possible list entries
        if dtype == string[:3]:                                 # Check if Arg type equals 1st 3 bytes of string
            if dtype == 'all':                                  # Select all columns
                list_columns = [clean[0] for clean in filename.dtypes.iteritems()]
            else:                                               # Select ONLY columns that are in the list
                list_columns = [clean[0] for clean in filename.dtypes.iteritems() if clean[1] == string]
            print ('Searching for', string, 'and UNIQUE values')# Print Heading
            for col in list_columns:                            # Read thru the list of field values
                print ('\n','='*10, col, '='*10)                # Print Column Name
                print (filename[col].unique(), '\n')            # Print the unique values in the Column    
    return 

       
def field_count(filename, dtype):
    list = ['all','int64','float','object']                     # Possible list entries
    dtype = dtype[:3].lower()                                   # Store Arg type as 3 byte lowercase
    for string in list:                                         # Read thru the possible list entries
        if dtype == string[:3]:                                 # Check if Arg type equals 1st 3 bytes of string
            if dtype == 'all':                                  # Select all columns
                list_columns = [clean[0] for clean in filename.dtypes.iteritems()]
            else:                                               # Select ONLY columns that are in the list
                list_columns = [clean[0] for clean in filename.dtypes.iteritems() if clean[1] == string]
            
            print ('\nFor', string, 'finding column field value counts\n', list_columns, '\n') # Print Heading
            hold_name = ' '                                     # Initialize holding variable
            for col in list_columns:                            # Read thru the list of field values
                if hold_name != col:                            # test to see if column has changed
                    hold_name = col                             # move column name to holding variable 
                    print ('\n','='*10, col, '='*10)            # Print Column Name
                print (filename[col].value_counts(), filename[col].unique())            # Print the value count
    return
      
    
def corr_heat_map(filename):

    fig, ax = plt.subplots(figsize=(45,15))                                      # Set the default matplotlib figure size:

    mask = np.zeros_like(filename.corr(), dtype=np.bool)                         # Generate a mask for the upper triangle
    mask[np.triu_indices_from(mask)] = True
    
    # Assign the matplotlib axis the function returns. This will let us resize the labels.
    ax = sns.heatmap(filename.corr(), mask=mask, annot=True)                     # Plot the heatmap with seaborn.

    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14, rotation=30)      # Resize X labels 
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14, rotation=0)       # Resize y labels

    plt.show()                                                                   # Display Correlated Heat map
    return    
    
    
def OLS_regression(filename, target):
    Y = filename[target]
    
    list_columns = [clean[0] for clean in filename.dtypes.iteritems() if clean[1] != 'object']  # Select all columns to a list
    
    list_columns.remove(target)                         # Drop the Target from the list of columns

    for col in list_columns:                            # Read thru the list of field values
        if filename[col].isna().count() > 0:
            filename[col] = filename[col].fillna(0)
#         else:
#             filename[col] = filename[col]
                                     
        if filename[col].isna().count() >= 0:
            print ('\n','='*10, col, '='*10)
            X = filename[col]
            X = sm.add_constant(X)
            model = sm.OLS(Y, X)
            results = model.fit()
#        results.summary()    
            print(results.summary())
        else:
            print ('THIS COLUMN HAS BEEN EXCLUDED  ==> ', col, '<===') 
    return 


def file_overview(filename):
    filename_info = pd.DataFrame()
    rows = float(filename.shape[0])
    print ('Number of ROWS-->', int(rows), 'Number of COLUMNS-->', filename.shape[1])
    for a,b in filename.items():
        msg,val = [],[ str(filename[a].dtype), filename[a].nunique(), filename[a].isnull().sum(),0] #Count [Unique,null]
        if (val[1] != rows): msg.append(str(val[1]) + ' Unique')
        if (val[2]):  msg.append(str(val[2]) + ' Null {0:.1f}%'.format(val[2] / rows * 100))
        if (val[0] == 'object'): #String / Object
            wierd = list(filename[a].str.match(r'^(~|`|!|@|#|\$|%|\^|&|\*|\(|\)|-|_|\+|=|\{|\}|\[|\]|\\|\||:|;|"|<|>|,|\.|\?|/|\s)+$')) #Match Cell with weird input
            if True in wierd: print ('WIERD ' + a + ' :', [str(c) + ': ' + str(wierd[c]) for c in range(len(wierd)) if wierd[c] is True]) #If Got Match
        else: #Other Dtypes
            val[3] = filename[filename[a] == 0].shape[0]
            if(val[3]>0):
                #df[a].replace(0,np.nan,inplace=True) #OPTIONAL: 0 ==> NaN
                msg.append(str(val[3])+' Zero {0:.1f}%'.format(val[3]/rows*100))
        filename_info = filename_info.append({'Column':str(a), 'Type' : val[0], 'Unique' : val[1], 'Null' : val[2], 'Null%' : round(val[2] / rows * 100, 1), 'Zero' : val[3], 'Zero%' : round(val[3] / rows * 100, 1)}, ignore_index=True)
        #if(len(msg)>1): print a+' : '+' | '.join(msg)
    return filename_info.sort_values('Null', ascending = False)


# Compare and see if NaN values are related with associated Columns 
def compare_column_nan(filename, column, comparing_column):
    filename_all = filename.loc[(filename[column].isnull()), comparing_column] # Get Only the columns containing NaN 
    column_diff = filename_all.size - filename_all.notna().size             
    if column_diff:                                                            # Check if the file sizes are different
        print(column, ':', column_diff, 'non-NaN don''t match')                # If file sizes differ then print 
    else: 
        print(column, 'All NaN''s are related with', comparing_column)         # Otherwise they match
        

def column_fillna_to(filename, column_list, value):
#Fill as NA
    for column in column_list:
        try: 
            filename[column].fillna(value,inplace=True)
        except: 
            pass        


# Function to crossvalidate accuracy of a knn model acros folds
def accuracy_crossvalidator(X, y, knn, cv_indices):
    
    # list to store the scores/accuracy of folds
    scores = []
    
    # iterate through the training and testing folds in cv_indices
    for train_i, test_i in cv_indices:
        
        # get the current X train & test subsets of X
        X_train = X[train_i, :]
        X_test = X[test_i, :]

        # get the Y train & test subsets of Y
        Y_train = y[train_i]
        Y_test = y[test_i]

        # fit the knn model on the training data
        knn.fit(X_train, Y_train)
        
        # get the accuracy predicting the testing data
        acc = knn.score(X_test, Y_test)
        scores.append(acc)
        
        print(('Fold accuracy:', acc))
        
    print(('Mean CV accuracy:', np.mean(scores)))
    return scores

#TRANSFORM NUMERICAL FEATURE TO NORMAL DISTRIBUTION
def normalize_data(filename, num_list, limit = .5, diff = .2):
    '''
    Function auto-select best transformation to normalize data (based on skewness)
    limit : Skew Limit
    diff : Min skew difference for transform
    '''
    name = ['Log','Log +1','Log 10','Square Root','Cube Root','Power 2','Exponential','Exponential 2','Exponential -1']
    print ('SKEWNESS TRANSFORM')
    for col in num_list:
        orig = round(np.abs(filename[col].skew()), 2)
        if(orig > limit):
            skew = []                                                          # List of Skewness after transform
            method = [np.log(filename[col]),   np.log1p(filename[col]),  
                      np.log10(filename[col]), np.sqrt(filename[col]), 
                      np.cbrt(filename[col]),  np.power(filename[col], 2),
                      np.exp(filename[col]),   np.exp2(filename[col]),
                      np.expm1(filename[col])]                                 # Methods to transform feature
            for a in method:                                                   # Calc skew for each method
                b = round(np.abs(a.skew()), 2)
                skew.append(9999.) if np.isnan(b) else skew.append(b)          # float if skew = NaN else Add skew
            mini = min(skew)                                                   # Get lowest skew
            if(orig-mini > diff):
                i = skew.index(mini)
                filename[col] = method[i]
                print (col, '\t', orig, '\t==>', mini, '\t', name[i], '\t', np.where(mini > limit, '\tskewed', '')) # Transformed Feature
            #else: print col,orig,' =/= ',mini,name[i],np.where(mini>limit,'skewed','') #Un-transform Feature Summary

            
#LASSO METHOD
from sklearn.linear_model import Lasso, LassoCV
def use_lasso(X, y, cols, folds = 5):
    #Find Optimal Alpha
    optimal = LassoCV(cv = folds, verbose = 1)
    optimal.fit(X, y)
    print ('LASSO\nOptimal Alpha:', optimal.alpha_)

    #Use Optimal Alpha
    model = Lasso(alpha = optimal.alpha_)
    scores = cross_val_score(model, X, y, cv = folds)
    score_avg = scores.mean()
    print ('Score: {:.3f} ({:.3f})\n'.format(score_avg, scores.std()))

    #Get Top 10 Features
    model.fit(X, y)
    top = [[a, b] for a, b in zip(cols, model.coef_)]
    print (pd.DataFrame(top, columns = ['Feature', 'Coefficient']).sort_values(by = 'Coefficient', ascending = False)[:10])
    return model, score_avg


#RIDGE METHOD
from sklearn.linear_model import Ridge, RidgeCV
def use_ridge(X, y, cols, folds = 5):
    #Find Optimal Alpha
    optimal = RidgeCV(alphas = np.logspace(-2, 7), cv = folds)
    optimal.fit(X, y)
    print ('RIDGE\nOptimal Alpha:', optimal.alpha_)

    #Use Optimal Alpha
    model = Ridge(alpha = optimal.alpha_)
    scores = cross_val_score(model, X, y, cv = folds)
    score_avg = scores.mean()
    print ('Score: {:.3f} ({:.3f})\n'.format(score_avg, scores.std()))

    #Get Top 10 Features
    model.fit(X, y)
    top = [[a, b] for a, b in zip(cols, model.coef_)]
    print (pd.DataFrame(top, columns = ['Feature', 'Coefficient']).sort_values(by = 'Coefficient', ascending = False)[:10])
    return model, score_avg


#ELASTIC-NET METHOD
def use_elastic(X, y, cols, folds = 5):
    #Find Optimal Alpha & L1
    optimal = ElasticNetCV(cv = folds)
    optimal.fit(X, y)
    print ('ELASTIC-NET\nOptimal Alpha:', optimal.alpha_, '\nL1 Ratio:', optimal.l1_ratio_)

    #Use Optimal Alpha & L1 Ratio
    model = ElasticNet(alpha = optimal.alpha_, l1_ratio = optimal.l1_ratio_)
    scores = cross_val_score(model, X, y, cv = folds)
    score_avg = scores.mean()
    print ('Score: {:.3f} ({:.3f})\n'.format(score_avg, scores.std()))

    #Get Top 10 Features
    model.fit(X, y)
    top=[[a, b] for a, b in zip(cols, model.coef_)]
    print (pd.DataFrame(top, columns = ['Feature', 'Coefficient']).sort_values(by = 'Coefficient', ascending=False)[:10])
    return model, score_avg


    # LASSO / RIDGE /ELASTIC-NET METHOD
def use_lasso_ridge_elastic(X_train, y_train, X_test, y_test,cols, min_score =.7, folds=10):
    print (X_train.shape[1], 'features\n')
    for Model in [use_lasso, use_ridge, use_elastic]:
        print (Model, '--', Model.__name__)
        model, score = Model(X_train, y_train, cols,folds)

        if(score < min_score): 
            print ('\nScore', score, '< Minimum', min_score)
        else: #If Model Mean Score >= min_score
            y_pred=cross_val_predict(model, X_test, y_test,cv = folds)
            print ('\ny-Predict\nAccuracy:', metrics.r2_score(y_test, y_pred), '\nMSE:', metrics.mean_squared_error(y_test, y_pred),'\n\n')


def quantitative_summarized(filename, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True, swarm=False):
    '''
    Helper function that gives a quick summary of quantattive data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data (usually the target variable)
    y: str. vertical axis to plot the quantitative data
    hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
    palette: array-like. Colour of the plot
    swarm: if swarm is set to True, a swarm plot would be overlayed
    Returns
    =======
    Quick Stats of the data and also the box plot of the distribution
    '''
    series = filename[y]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.boxplot(x=x, y=y, hue=hue, data=filename, palette=palette, ax=ax)

    if swarm:
        sns.swarmplot(x=x, y=y, hue=hue, data=filename, palette=palette, ax=ax)

    plt.show()
    return

def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True):
    '''
    Helper function that gives a quick summary of a given column of categorical data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data, y would be the count
    y: str. vertical axis to plot the labels of categorical data, x would be the count
    hue: str. if you want to compare it another variable (usually the target variable)
    palette: array-like. Colour of the plot|
    Returns
    =======
    Quick Stats of the data and also the count plot
    '''
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = dataframe[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette, ax=ax)
    plt.show()
