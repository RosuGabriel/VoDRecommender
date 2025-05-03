#%%
# Imports
from data_loader.data_movie_lens import MovieLensData
import pandas as pd
from recommenders.hybrid_filtering import HybridRecommender
import time
from sklearn.metrics import mean_squared_error
import math
import random



# Function to split ratings into train and test dfs
def split_train_test_ratings(ratingsDf, trainSize=0.8):
    # Function to split ratings from group
    def _split_user_group(group):
        train = group.sample(frac=trainSize)
        test = group.drop(train.index)
        return train, test

    trainList = []
    testList = []

    # For each user take a train and test sample
    for _, group in ratingsDf.groupby('userId'):
        train, test = _split_user_group(group)
        trainList.append(train)
        testList.append(test)

    # Combine rows
    trainDf = pd.concat(trainList).reset_index(drop=True)
    testDf = pd.concat(testList).reset_index(drop=True)

    return trainDf, testDf


#%%
# Load data
startTime = time.time()

data = MovieLensData(includeMyRatings=True)

endTime = time.time()
print(f"Exec time: {endTime - startTime} s")

#%%
# Replace original ratingsDf with trainDf, create recommender and create testDf
startTime = time.time()

data.ratingsDf, testDf = split_train_test_ratings(data.ratingsDf, trainSize=0.8)

# Collaborative model removes some movies for better sparsity
recommender = HybridRecommender(data=data, freshModel=True)

data.mergeRatingsWithFeatures()
data.data_sparsity()

# Test eliminats the movies that are not in the training set
testDf = testDf[testDf['movieId'].isin(data.ratingsDf['movieId'])] # Keep movies existent in trainDf (some are removed for sparsity)

# Gruppping testDf by userId
groupedTest = testDf.groupby('userId')

# List of all users in test
users = testDf['userId'].unique()

endTime = time.time()


#%%
# Performance test
totalRmse = 0
rounds = 5
for round in range(rounds):
    print(f"Round {round+1}")
    usersSample = random.sample(list(users), int(len(users)/40))
    len(usersSample)

    startTime = time.time()

    # Predict test movies for the first user
    predictions = []
    trueRatings = []
    firstIteration = True

    for user in usersSample:
        # print("User:", user)
        if firstIteration:
            recommender.initialize_recommender(user, collaborativeWeight=0.4, disagreementPenalty=0.7, titleWeight=1, ratingsWeights=[1.2,1,0.6,-0.7,-2], genresWeights=[2], )
            firstIteration = False
        else:
            recommender.change_user(user)

        user_group = groupedTest.get_group(user)

        for idx, movie in enumerate(user_group['movieId'].values):
            # print("Movie:", movie)
            result = recommender.get_movie_predicted_rating(movie)
            
            if not result.empty and 'hybridRating' in result:
                pred = result['hybridRating'].values[0]
                # print("Prediction:", pred)
                predictions.append(pred)
                trueRatings.append(user_group.iloc[idx]['rating'])
            # else:
            #     print(f"WARNING: No prediction for User {user}, Movie {movie}")

    mse = mean_squared_error(trueRatings, predictions)
    rmse = math.sqrt(mse)
    print(f"RMSE: {rmse}")
    totalRmse += rmse


print(f"\nAvg RMSE: {totalRmse/rounds}")
endTime = time.time()
print(f"Exec time: {endTime - startTime} s")
