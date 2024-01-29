from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix
import pandas as pd


class Base_SearchCV:
    """ A class object that stores attributes to be used as parameters \
in a cross validation search object.
    
    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Pipeline to be used in SearchCV as the estimator
    
    scorer : sklearn.metrics._scorer._PredictScorer
        Scoring function to use in SearchCV object
     
     cv : int, sklearn.model_selection._split
         Number of folds to carry out, or type and number of folds \
to carry out passed as a cross validator object.
    
    X_train : array-like, shape of (n_examples, n_features)
        Train set features matrix.
    
    y_train : array-like, shape of (n_features,)
        Train set target array.
    
    Returns
    -------
    None
         """
    def __init__(self, pipe, scorer, cv, X_train, y_train):
        self.pipe = pipe
        self.scorer = scorer
        self.cv = cv
        self.X_train = X_train
        self.y_train = y_train

    def grid_search(self, search_space):
        """ Returns a GridSearchCV object fitted on hyperparameters \
provided in search_space dictionary. Uses attributes of base_SearchCV \
class object to pass as parameters to GridSearchCV.

        Parameters
        ---------
        search_space : dict
            A dictionary containing hyperparameter keys and desired\
search values.
        
        Returns
        -------
        grid_fitted : sklearn.model_selection._search.GridSearchCV
            A fitted GridSearchCV object.
        """
        grid = GridSearchCV(
            self.pipe,
            search_space,
            cv=self.cv,
            scoring=self.scorer,
            return_train_score=True,
            n_jobs=-1,
        )
        grid_fitted = grid.fit(self.X_train, self.y_train)
        return grid_fitted

    def random_search(self, search_space, n_iter):
        """Returns a RandomizedSearchCV object fitted on hyperparameters \
provided in search_space dictionary after n iterations specified. Uses \
attributes of base_SearchCV class object to pass as parameters to RandomizedSearchCV.
        
        Parameters
        ---------
        search_space : dict
            A dictionary containing hyperparameter keys and desired \
 search values or ranges.
 
         n_iter : int
             The number of random iterations desired.
             
        Returns
        -------
        random_fitted : sklearn.model_selection._search.RandomizedSearchCV 
            A fitted RandomizedSearchCV object.
             """
        # Create randomized search object
        random = RandomizedSearchCV(
            self.pipe,
            search_space,
            cv=self.cv,
            scoring=self.scorer,
            return_train_score=True,
            n_jobs=-1,
            n_iter=n_iter,
            random_state =777,
        )
        # Fit randomized search object on X_train and y_train
        random_fitted = random.fit(self.X_train, self.y_train)
        # Return fitted object
        return random_fitted

def print_best_validation(model):
    
    """ 
    Prints the best validation score and related hyperparameters \
of a fitted SearchCV object.

    Parameters
    ---------
    model : sklearn.model_selection._search.RandomizedSearchCV, \
sklearn.model_selection._search.GridSearchCV    
        A fitted SearchCV object.
    
    Returns
    -------
    None
"""
    print("Parameters with Best validation Score:")
    print(model.best_params_)
    print()
    print("Best Validation Score:")
    print(model.best_score_)
    
def evaluate_best_model(model,X_test,y_test):
    
    """ 
    Prints the classification report of the passed test feature \
matrix and test target array and plots the related confusion matrix.
    
    Parameters
    ----------
    model : sklearn.model_selection._search.RandomizedSearchCV \
sklearn.model_selection._search.GridSearchCV  
        A fitted SearchCV object.  

    X_test : array-like, shape of (n_examples, n_features)
        Test set feature matrix.
        
    y_test : array-like, shape of (n_features,)
        Test set target array.
    
    Returns
    -------
    confusion_matrix : sklearn.metrics._plot.confusion_matrix.\
ConfusionMatrixDisplay
        The confusion matrix for the passed fitted model, X_test and \
y_test values.
    """       
    print("Test Set Classification Report:")
    print(classification_report(y_test,model.predict(X_test)))
    print()
    print("Test Set Confusion Matrix:")
    confusion_matrix = plot_confusion_matrix(model,X_test,y_test)
    return confusion_matrix

def cv_score_df(model, hparams, n_rows):
    """
    Creates a dataframe of cross validation scores and related \
hyperparameters and returns top n_rows.

    Parameters
    ----------
     model : sklearn.model_selection._search.RandomizedSearchCV \
sklearn.model_selection._search.GridSearchCV
        A fitted SearchCV object.
     
     hparams : str, list(str)
         A string or list of strings referencing desired hyperparameter keys.
     
     n_rows : int
         The number of best scores to populate to the dataframe.
     
     Returns
     -------
     sorted_full_df : pd.DataFrame
         A sorted dataframe consisting of n_rows of top validation \
         scores and their associated hyperparameters                  
    """
    # Create df with classifier name and scores
    score_df = pd.DataFrame(model.cv_results_).loc[:,['mean_train_score','mean_test_score']]
    # Create df with hyperparameter values
    hparam_df = pd.DataFrame(model.cv_results_).loc[:,hparams]
    # Concatenate scores and hyperparameters
    full_df = pd.concat([hparam_df,score_df],axis=1)
    # Sort full dataframe, trim to top n rows
    sorted_full_df = full_df.sort_values(by='mean_test_score',ascending=False).head(n_rows)
    #Return sorted and trimmed df
    return sorted_full_df
       
    