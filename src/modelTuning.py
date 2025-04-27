from data import MovieLensData
from surprise import Dataset, Reader, SVD, BaselineOnly, accuracy
from surprise.model_selection import RandomizedSearchCV

#%%
# Get dataset
data = MovieLensData(includeMyRatings=True)

#%%
# Convert to Surprise dataset
reader = Reader(rating_scale=[0.5, 5])
ratingsData = Dataset.load_from_df(data.ratingsDf[['userId', 'movieId', 'rating']], reader)
trainset = ratingsData.build_full_trainset()


#%%
# Params for tuning
SVDParams = {
    'n_epochs': [2000, 1000, 3000],
    'lr_all': [0.002, 0.001, 0.003],
    'reg_all': [0.1, 0.2, 0.05],
    'n_factors': [100, 80, 150]
}

BOParams = {
    'bsl_options': {
    'method': ['sgd', 'als'],
    'learning_rate': [0.015, 0.02, 0.01],
    'n_epochs': [110, 90, 1000],
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
# SVD model training
model = SVD(n_epochs=2000, lr_all=0.001, reg_all=0.1, n_factors=100)

#%%
# Model evaluation on a sintetic test set
predictions = model.test(trainset.build_anti_testset())
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
