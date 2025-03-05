from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import math
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import statistics
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Define a function scraping all links to competitions within a given month
def scrape_month(year, month, verbose=False):
    """
    Function scraping links to all competitions within a given month
    """
    
    # Get the html code
    link = f'https://www.csts.cz/cs/VysledkySoutezi/Souteze?rok={year}&mesic={month}'
    req = requests.get(link)
    if req.status_code != 200:
        raise Exception(f'Request unsuccessful for link {link}. Status code: {req.status_code}')
    soup = BeautifulSoup(req.content, 'html.parser')

    # Get links to all competitions
    ems = soup.find_all('em') # Competitions are in italic
    a = [i.find('a') for i in ems] # Get a tags containg the competition category and link
    cats = [i.text for i in a] # Get competition category
    links = ['https://www.csts.cz' + i['href'] for i in a] # Get the link to the competition

    # Check if the number of competitions corresponds to the number stated on the website
    divs = soup.find_all('div')
    check = int(re.search('měsíc: ([0-9]*)',divs[9].text).groups()[0])
    if check != len(cats):
        print(f'For year {year} and month {month} the expected number of competitions is {check} but {len(cats)} was scraped')
    elif verbose:
        text = f'Scraped {check} competitions for year {year} and month {month}'
        print(text)
        with open('all_links.txt', 'a') as out:
            out.write(text + '\n')

    return list(zip(cats, links))

# Function scraping the results of the given competition based on a link
def scrape_competition(link):
    """
    Function scraping the results of the given competition based on a link
    """
    # Get the soup
    req = requests.get(link)
    if req.status_code == 200:
        soup = BeautifulSoup(req.content, 'html.parser')
    else:
        return f'Unsuccessful request for {link}. Status code: {req.status_code}'
    
    # Get the competition title
    title = soup.find('h2').text

    # Get the results
    res = pd.read_html(link, encoding='utf-8')

    return title, res[0].to_dict(), res[1].to_dict()

# Function checking that a value represents a rank
def validate_rank(val):
    """
    Function checking that a value represent a rank
    """
    if '-' in val or val.isnumeric(): # Check if it is a split rank or a normal rank
        return True
    else:
        return False

# Function processing raw competition data
def process_comp(comp_data):
    """
    Function processing raw competition data
    """
    # Convert results to pandas
    res = pd.DataFrame(comp_data[0])

    # Check for invalid columns
    if 'Unnamed: 5' in res.columns:
        return 'Without points'

    # Check number of columns
    if res.shape[1] != 8:
        return f'Invalid number of columns: {res.shape[1]}. Column names: {list(res.columns)}'
    
    # Filter out invalid rows
    try:
        res = res.loc[res['Umístění'].apply(validate_rank), :]
    except KeyError:
        return f'Column Umístění not in the data. Actual name: {res.columns[1]}'

    # Drop invalid columns
    for col_idx, col in zip([1, 6, 7], ['Číslo páru', 'Nově získaná třída', 'Pozn.']): # Loop through columns to drop
        try:
            res.drop(col, axis=1, inplace=True)
        except KeyError:
            return f'Column {col} not in the data. Actual name: {res.columns[col_idx]}'

    # Add new columns
    res['N'] = res.shape[0]
    res['Judges'] = comp_data[1][1][0]

    return res

# Function converting ranks to numbers
def convert_rank(rank):
    """
    Function converting ranks to numbers
    """
    try: # Try converting
        return int(rank)
    except ValueError: # Handle split ranks
        return int(rank.split(' - ')[0]) # Get the first of the split ranks
    
# Function checking whether the given participant should receive a final or not
def is_final(row):
    """
    Function checking whether the given participant should receive a final or not
    """
    # Non-finalists are not eligible
    if row['Finalist'] == 0: 
        return 0 
    
    # Define auxiliary variables
    rank = row['Rank'] # Get the rank
    n = row['N'] # Get the number of participants
    unique_clubs = row['Unique clubs'] # Get the number of unique clubs
    bounds = pd.DataFrame({'Min':[1, 3, 6, 11, 16, 21, 25], 'Max':[2, 5, 10, 15, 20, 25, math.inf],
                            'Required position':[-math.inf, 1, 2, 3, 4, 5, math.inf], 'Unique clubs':[1, 2, 3, 3, 3, 3, 3]})
    
    # Check requirements
    requirements = bounds.loc[(bounds['Min'] <= n) & (n <= bounds['Max']), ['Required position', 'Unique clubs']].values[0]
    if (rank <= requirements[0]) and (unique_clubs >= requirements[1]): # If the requirements are satisfied
        return 1
    else:
        return 0

# Function exporting a given data set
def export_dataset(df, name, kind='xlsx', overwrite='No', verbose=True):
    """
    Function exporting a given data set
    """
    # Print info
    if verbose: print(f'{name.title()} data set - Total rows: {df.shape[0]:,}. Total columns: {df.shape[1]}'.replace(',', ' '))
    
    # Check if a file already exists
    if overwrite != 'Yes':
        if os.path.isfile(f'data/{name}_data_set.{kind}'): # Check if the file exists
            overwrite = input(f'Rewrite the existing {name} data set? ')
        else:
            overwrite = 'Yes'

    # Store the file
    if overwrite == 'Yes':
        if kind == 'xlsx':
            df.to_excel(f'data/{name}_data_set.xlsx', index=False)
        else:
            df.to_csv(f'data/{name}_data_set.csv', index=False, encoding='utf-8', sep=';')
    else:
        print(f'{name.title()} data set was not stored.')

# Function plotting a histogram of a univariate variable
def plot_univariate(var, cat=False, out=False):
    """
    Function plotting a histogram of a univariate variable
    """
    if cat:
        var.value_counts()[:5].plot.bar()
        plt.title(var.name)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        var.hist(ax=ax[0])
        add = 1 if 0 in var else 0
        np.log(var + add).hist(ax=ax[1])
        ax[1].set_title(f'{var.name} (log)')
        ax[0].set_title(var.name)
    if out:
        plt.savefig(out)
    else:
        plt.show()

# Function estimating a univariate logistic regression
def log_reg_univariate(var, dep, log=False, squared=False, robust=True, verbose=False):
    """
    Function estimating a univariate logistic regression
    """
    # Convert to logarithms if desired
    if log:
        var = np.log(var + int(0 in var))

    # Add a squared term if required
    if squared:
        var = pd.concat([var, var ** 2], axis=1)

    # Add a constant
    var = sm.add_constant(var)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(var, dep, test_size=0.2, random_state=123)

    # Estimate
    if robust: # If robust covariance estimation is required
        log_reg = sm.Logit(y_train, X_train).fit(cov_type='hc3', disp=verbose)
    else:
        log_reg = sm.Logit(y_train, X_train).fit(disp=verbose)

    # Extract statistics
    preds = log_reg.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    mse = statistics.mean((preds - y_test)) ** 2
    pvals = log_reg.pvalues
    pvals.drop('const', inplace=True)
    if len(pvals) == 1:
        pvals = pvals.values[0]
    else:
        pvals = pvals.values

    # Print info
    if verbose:
        print(f'AUC (test): {auc:.2%}')
        print(f'MSE (test): {mse:.2%}')
        print(log_reg.summary())

    return auc, mse, log_reg.prsquared, pvals

class NumericalTransformer(BaseEstimator, TransformerMixin):
    """
    Class converting features to logarithms, creating polynomial features, and standardizing
    """
    def fit(self, X, y):
        """
        Fitting the transformer
        """
        X = np.log(X + 1)
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        X = self.poly.fit_transform(X)
        self.scaler = StandardScaler()
        self.scaler.fit(X)

        return self
    
    def transform(self, X):
        """
        Function transforming given data
        """
        X = np.log(X + 1)
        X = self.poly.transform(X)
        return self.scaler.transform(X)


# Function creating a pipeline including preprocessing and estimation
def make_pipeline(model, num_vars, cat_vars=None, poly=False):
    """
    Function creating a pipeline including preprocessing and estimation
    """

    # Define transformers
    scaler = StandardScaler()
    log_transform = FunctionTransformer(lambda x: np.log(x + 1))

    # Create the pipeline
    if cat_vars:
        one_hot = OneHotEncoder(handle_unknown='ignore')
        if poly:
            num_transf = NumericalTransformer()
            pipe = Pipeline([('preprocessing', ColumnTransformer([('numerical', num_transf, num_vars), ('one_hot', one_hot, cat_vars)])),
                              ('model', model)])
        else:
            num_vars_new, cat_vars_new = list(range(len(num_vars))), list(range(len(num_vars), len(num_vars) + len(cat_vars)))
            pipe = Pipeline([('log_transform', ColumnTransformer([('log_tranform', log_transform, num_vars), ('pass', 'passthrough', cat_vars)])),
                            ('scale_and_hot', ColumnTransformer([('scaler', scaler, num_vars_new), ('one_hot', one_hot, cat_vars_new)])),
                            ('model', model)])
    else:
        if poly:
            pipe = Pipeline([('preprocessing', ColumnTransformer([('numerical', NumericalTransformer(), num_vars)], remainder='passthrough')),
                              ('model', model)])
        else:
            pipe = Pipeline([('log_transform', ColumnTransformer([('log_tranform', log_transform, num_vars)], remainder='passthrough')),
                          ('scaler', ColumnTransformer([('scaler', scaler, num_vars)], remainder='passthrough')), ('model', model)])
        
    return pipe


