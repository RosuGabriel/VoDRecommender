#%%
# Imports
from data import MovieLensData
from surprise import Dataset, Reader, SVD

#%%
# Constants
model = SVD(n_epochs=2000, lr_all=0.001, reg_all=0.1, n_factors=100)

# Functions
def train_recommender(data: MovieLensData, model):
    # Reduce data size for smaller sparsity
    print("!Some movies will be eliminated in order to reduce sparsity!")
    print(f"Sparsity: {data.data_sparsity()}% ->", end=' ')
    data.remove_movies_with_less_than_k_ratings(10)
    print(f"{data.data_sparsity()}%")

    # Data converting to Surprise
    reader = Reader(rating_scale=[0.5, 5])
    ratingsData = Dataset.load_from_df(data.ratingsDf[['userId', 'movieId', 'rating']], reader)
    trainset = ratingsData.build_full_trainset()
    
    # Model training
    model.fit(trainset)

    return model

def predict_rating(row, model, userId: int):
    prediction = model.predict(userId, row['movieId'])
    return prediction.est

def get_recommendations(userId: int, data: MovieLensData, model, recomandationsNum: int=-1):
    userRatings = data.mergedDf[data.mergedDf['userId'] == userId].dropna()

    notWatchedMovies = data.moviesFeatures[~data.moviesFeatures['movieId'].isin(userRatings['movieId'])].dropna() # "~" is the negation operator for a boolean mask
    notWatchedMovies['predictedRating'] = notWatchedMovies.apply(lambda row: predict_rating(row=row, model=model, userId=userId), axis=1)
    return notWatchedMovies.sort_values(by='predictedRating', ascending=False)[:recomandationsNum]

#%%
# # Usage example
# # Get data and model
# data = MovieLensData(includeMyRatings=True)
# trainedModel = train_recommender(data, model)
# #%%
# # Get recommendations
# get_recommendations(611, data, trainedModel, recomandationsNum=20)