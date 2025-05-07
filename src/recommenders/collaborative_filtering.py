#%%
# Imports
from data_loader.movie_lens_data import MovieLensData
from surprise import Dataset, Reader, SVD
import joblib
from utils.paths import MODELS_DIR



# Functions
def train_recommender(data: MovieLensData, model, newModelName: str=None):
    # Reduce data size for smaller sparsity
    print("Movies num:",data.moviesDf.shape)[0]
    if data.data_sparsity() > 95:
        print("!Some movies will be eliminated in order to reduce sparsity!")
        print(f"Sparsity: {data.data_sparsity()}% ->", end=' ')
        data.remove_movies_with_less_than_k_ratings(10)
        print(f"{data.data_sparsity()}%")
    print("Movies num:",data.moviesDf.shape)[0]
    
    # Data converting to Surprise
    reader = Reader(rating_scale=[0.5, 5])
    ratingsData = Dataset.load_from_df(data.ratingsDf[['userId', 'movieId', 'rating']], reader)
    trainset = ratingsData.build_full_trainset()
    
    # Model training
    model.fit(trainset)
    if newModelName is not None:
        joblib.dump(model, MODELS_DIR / f"{newModelName}.pkl")

    return model


def import_model(modelName: str='SVD model'):
    if modelName is None:
        modelName = 'SVD model'
    return joblib.load(MODELS_DIR / f"{modelName}.pkl")


def _predict_rating(row, model, userId: int):
    return model.predict(userId, row['movieId']).est


def get_recommendations(data: MovieLensData, userId: int, model, recomandationsNum: int=-1):
    userRatings = data.user_ratings(userId=userId).copy()

    notWatchedMovies = data.moviesFeatures[~data.moviesFeatures['movieId'].isin(userRatings['movieId'])].dropna() # "~" is the negation operator for a boolean mask
    notWatchedMovies['predictedRatingCF'] = notWatchedMovies.apply(lambda row: _predict_rating(row=row, model=model, userId=userId), axis=1)
    return notWatchedMovies.sort_values(by='predictedRatingCF', ascending=False)[:recomandationsNum]



# #%%
# # Usage example
# # Get data and model
# data = MovieLensData(includeMyRatings=True)
# trainedModel = train_recommender(data=data, model=SVD(n_epochs=2000, lr_all=0.001, reg_all=0.1, n_factors=100))
# #%%

# # Get recommendations
# get_recommendations(data, 611, trainedModel, recomandationsNum=20)