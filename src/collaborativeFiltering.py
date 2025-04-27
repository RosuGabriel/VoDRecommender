#%%
# Imports
from data import MovieLensData
from surprise import Dataset, Reader, SVD
import joblib



# Functions
def train_recommender(data: MovieLensData, model, newModelName: str=None):
    # Reduce data size for smaller sparsity
    if data.data_sparsity() > 95:
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
    if newModelName is not None:
        joblib.dump(model, f'{newModelName}.pkl')

    return model


def import_model(modelName: str='SVD model.pkl'):
    if modelName is None:
        return joblib.load(f"SVD model.pkl")
    return joblib.load(modelName)


def _predict_rating(row, model, userId: int):
    return model.predict(userId, row['movieId']).est


def get_recommendations(data: MovieLensData, userId: int, model, recomandationsNum: int=-1):
    userRatings = data.mergedDf[data.mergedDf['userId'] == userId].dropna()

    notWatchedMovies = data.moviesFeatures[~data.moviesFeatures['movieId'].isin(userRatings['movieId'])].dropna() # "~" is the negation operator for a boolean mask
    notWatchedMovies['predictedRatingCF'] = notWatchedMovies.apply(lambda row: _predict_rating(row=row, model=model, userId=userId), axis=1)
    return notWatchedMovies.sort_values(by='predictedRatingCF', ascending=False)[:recomandationsNum]



#%%
# # Usage example
# # Get data and model
# data = MovieLensData(includeMyRatings=True)
# trainedModel = train_recommender(data, model)
# #%%
# # Get recommendations
# get_recommendations(611, data, trainedModel, recomandationsNum=20)