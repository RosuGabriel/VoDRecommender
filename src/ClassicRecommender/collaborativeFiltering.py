#%%
# Imports
from data import Data
from surprise import Dataset, Reader, SVD, BaselineOnly, SVDpp
from surprise.model_selection import cross_validate, RandomizedSearchCV

#%%
# Get data object
data = Data()

#%%
# Reduce data size for smaller sparsity
data.remove_movies_with_less_than_k_ratings(10)
data.data_sparsity()

#%%
# Data converting to Surprise
reader = Reader(rating_scale=(data.ratingsDf.rating.min(), data.ratingsDf.rating.max()))
ratingsData = Dataset.load_from_df(data.ratingsDf[['userId', 'movieId', 'rating']], reader)

#%%
# Params for tuning
SVDParams = {
    'n_epochs': [50, 200],
    'lr_all': [0.005, 0.015, 0.01],
    'reg_all': [0.15, 0.05, 0.1],
    'n_factors': [50, 60, 40]
}

BOParams = {
    'bsl_options': {
    'method': ['sgd', 'als'],
    'learning_rate': [0.015, 0.02, 0.01],
    'n_epochs': [110, 90, 100],
    'reg': [0.04, 0.05, 0.03]
    }
}

#%%
# SVD tuning
rs = RandomizedSearchCV(SVD, SVDParams, measures=['rmse', 'mae'], cv=5, n_jobs=-1, joblib_verbose=2)
rs.fit(ratingsData)

#%%
# BaselineOnly tuning
rs = RandomizedSearchCV(BaselineOnly, BOParams, measures=['rmse', 'mae'], cv=5, n_jobs=-1, joblib_verbose=2)
rs.fit(ratingsData)

#%%
# Tuning results
print("Best RMSE:", rs.best_score['rmse'])
print("Best params:", rs.best_params['rmse'])

#%%
# SVD model
model = SVD(n_epochs=200, lr_all=0.015, reg_all=0.1, n_factors=50)
trainset = ratingsData.build_full_trainset()
model.fit(trainset)

#%%
# Ratings of a user
x = data.user_ratings(56)
x = zip(map(lambda y: data.title_of_movie(y[0]), x), x)
list(x)

#%%
# Predict ratings for a user
# All the items
allMovies = trainset.all_items()
allMoviesIds = [trainset.to_raw_iid(movieId) for movieId in allMovies]

# Items rated by the user
userId = 56
ratedMoviesIds = set([j for (j, _) in trainset.ur[trainset.to_inner_uid(userId)]])

# Unrated items
unratedMovies = [movieId for movieId in allMoviesIds if trainset.to_inner_iid(movieId) not in ratedMoviesIds]

# Predict unrated items
predictions = [model.predict(userId, movieId) for movieId in unratedMovies]

# Sort and get top 10
topN = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]

# Show titles
for prediction in topN:
    print(f"Item: {data.title_of_movie(prediction.iid)} Id: {prediction.iid}, Estimare: {prediction.est:.2f}")
