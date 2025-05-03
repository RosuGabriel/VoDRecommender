# Imports
from data_loader.data_movie_lens import MovieLensData
from content_based_filtering import get_recommendations as content_based_recommender
from collaborative_filtering import train_recommender, import_model, get_recommendations as collaborative_recommender
import pandas as pd
from surprise import SVD
from sklearn.preprocessing import MinMaxScaler



# Recommender class
class HybridRecommender:
    def __init__(self, data: MovieLensData=None, model=SVD(n_epochs=2000, lr_all=0.001, reg_all=0.1, n_factors=100),
                includeMyRatings: bool=True, freshModel: bool=False, modelName: str=None, disagreementPenalty: float=0.5,
                collaborativeWeight: float=0.5, includeFeatures: bool=False, userProfileFromCSV: str=None,
                ratingsWeights: list=[1,1,1,1,1], titleWeight: int=2, genresWeights: list=[1], yearWeight: float=1.0):
        
        self.data = data
        self.collaborativeWeight = collaborativeWeight
        self.disagreementPenalty = disagreementPenalty
        self.ratingsWeights = ratingsWeights
        self.titleWeight = titleWeight
        self.genresWeights = genresWeights
        self.yearWeight = yearWeight
        self.includeFeatures = includeFeatures
        self.userProfileFromCSV = userProfileFromCSV

        if self.data is None:
            self.data = MovieLensData(includeMyRatings=includeMyRatings)      
        
        if freshModel:
            self.model = train_recommender(self.data, model, modelName)
        else:
            self.model = import_model(modelName)

       
    def _make_hybrid_recommendations(self):
        # Merge recommendations
        hybridRecommendations = pd.merge(self.cbfRecommendations, self.cfRecommendations, on=['movieId','title','year'], how='inner')
        contentBasedWeight = 1 - self.collaborativeWeight
        
        # Wheighted average of predicted ratings
        hybridRecommendations['hybridRating'] = hybridRecommendations['predictedRatingCF']*self.collaborativeWeight + hybridRecommendations['predictedRatingCBF']*contentBasedWeight - self.disagreementPenalty * abs(hybridRecommendations['predictedRatingCF'] - hybridRecommendations['predictedRatingCBF'])
        ratingScaler = MinMaxScaler(feature_range=(0.5, 5))
        hybridRecommendations['hybridRating'] = ratingScaler.fit_transform(hybridRecommendations[['hybridRating']])
        hybridRecommendations.sort_values(by='hybridRating', ascending=False, inplace=True)
        self.hybridRecommendations = hybridRecommendations.reset_index(drop=True)


    def choose_user(self, userId: int):
        self.userId = userId

        # Remake recommendations for the new user
        if self.includeFeatures:
            self.cfRecommendations = collaborative_recommender(userId=self.userId, data=self.data, model=self.model, recomandationsNum=-1)
            self.cbfRecommendations = content_based_recommender(userId=self.userId, data=self.data, userProfileFromCSV=self.userProfileFromCSV, ratingsWeights=self.ratingsWeights,
                                                                titleWeight=self.titleWeight, genresWeights=self.genresWeights, yearWeight=self.yearWeight, recomandationsNum=-1)
        else:
            self.cfRecommendations = collaborative_recommender(userId=self.userId, data=self.data, model=self.model, recomandationsNum=-1)[['movieId', 'title', 'year', 'predictedRatingCF']]
            self.cbfRecommendations = content_based_recommender(userId=self.userId, data=self.data, userProfileFromCSV=self.userProfileFromCSV, ratingsWeights=self.ratingsWeights,
                                                                titleWeight=self.titleWeight, genresWeights=self.genresWeights, yearWeight=self.yearWeight, recomandationsNum=-1)[['movieId', 'title', 'year', 'predictedRatingCBF']]

        self._make_hybrid_recommendations()


    def user_real_ratings(self):
        return self.data.user_ratings(self.userId)


    def get_top_n_recommendations(self, n: int=-1):
        return self.hybridRecommendations[:n]


    def get_movie_predicted_rating(self, movieId: int, printMovie: bool=False):
        if printMovie:
            print('Movie:', self.data.title_of_movie(movieId))
        return self.hybridRecommendations[self.hybridRecommendations['movieId'] == movieId][['predictedRatingCF','predictedRatingCBF','hybridRating']]


    def get_recommendations_sample(self, size: int=20, minRating: float=4.0):
        return self.hybridRecommendations[self.hybridRecommendations['hybridRating'] >= minRating].dropna().sample(size)
