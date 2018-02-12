# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 15:10:59 2018

@author: zhenl
"""

import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns
import numpy as np
import scipy

def coficostfunc( params, Y, R, n_movies, n_features, n_users, lambda_ ):
    X, Theta = unfoldparams( n_movies, n_features, n_users, params );
    J = ( np.sum( ( np.dot( X, Theta.T ) * R - Y * R ) ** 2 ) /2 + 
         np.sum( Theta**2 ) * lambda_ / 2 + np.sum( X **2 ) * lambda_ / 2 )
    grad_X = ( np.dot( ( np.dot( X, Theta.T ) * R - Y * R ), Theta ) + 
              lambda_ * X )
    grad_Theta = ( np.dot( ( np.dot( X, Theta.T ) * R - Y * R ).T, X ) + 
                  lambda_ * Theta )
    grad_params = np.concatenate( 
            (grad_X.flatten(), grad_Theta.flatten()), axis=0 )
    return( J, grad_params );

def normalizeratings( Y, R ):
    Y_sum = np.sum( Y, axis = 1 )
    R_sum = np.sum( R, axis = 1 )
    Y_mean = Y_sum / R_sum
    Y_norm = np.zeros( Y.shape )
    for i in range( Y.shape[0] ):        
        j, = np.where( R[i,:] == 1 )
        Y_norm[i,j] = Y[i,j] - Y_mean[i]
    return( Y_norm, Y_mean ); 

def initparams( n_movies, n_features, n_users ):
    X = np.random.randn( n_movies, n_features )
    Theta = np.random.randn( n_users, n_features )
    init_params = np.concatenate( (X.flatten(), Theta.flatten()), axis=0 )
    return( init_params );
    
def unfoldparams( n_movies, n_features, n_users, theta ):
    X = theta[0:n_movies * n_features].reshape( n_movies, n_features )
    Theta = theta[n_movies * n_features::].reshape( n_users, n_features )
    return( X, Theta );

if __name__ == "__main__":
    df_mov = pd.read_csv( 'movies.csv' )
    df_small = pd.read_csv('ratings_small.csv').drop('timestamp', axis=1)
    df = df_small.pivot(index='movieId', columns='userId', values='rating')
    df = df.fillna(0)
    df.index.name = 'Movie ID'
    df.columns.name = 'User ID'
    Y = df.as_matrix()
    R = ( Y > 0. ).astype(int)
    n_movies, n_users = Y.shape
    Y_norm, Y_mean = normalizeratings( Y, R );
    n_features = 10
    
    init_params = initparams( n_movies, n_features, n_users );
    
    theta_ = scipy.optimize.minimize(
            fun = coficostfunc,
            x0 = init_params,
            args = (Y_norm, R, n_movies, n_features, n_users, 10 ),
            method = 'CG',
            jac = True,
            options = {
                    'maxiter': 100,
                    'disp': False,}).x
    
    X, Theta = unfoldparams( n_movies, n_features, n_users, theta_ )
    my_predictions = np.dot( X, Theta.T )[:,0] + Y_mean
    df_pred = pd.DataFrame( my_predictions, columns=['Predicted Rating'] )
    df_p = df_mov.join(df_pred)
    
        
    """
    plt.figure(figsize=(10,10))
    sns.heatmap(df, annot=False, fmt="d")
    plt.savefig('fig_heatmap.png',bbox_inches='tight')
    plt.show()

    '''
    check movie information, display # of movies in each genre
    '''
    df = pd.read_csv('movies_metadata.csv')
    df['genres'] = df['genres'].fillna('[]').apply(ast.literal_eval).apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else [] )

    df_g = df.apply(
            lambda x: pd.Series(x['genres']),axis='columns').stack().reset_index(
                    level=1, drop=True)
    df_g.name = 'genre'

    df_gen = df.drop('genres', axis=1).join(df_g)  
    df_gen = pd.DataFrame(df_gen['genre'].value_counts()).reset_index()
    df_gen.columns = ['genre', 'movie counts']
    plt.figure(figsize=(20,5))
    sns.barplot(x='genre', y='movie counts', data=df_gen.head(20))
    plt.savefig('fig_genre_counts.png',bbox_inches='tight')
    plt.show()
    """
