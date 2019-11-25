import pandas as pd
import numpy as np

## Create movie data vector based on embeddings or without.
## genre data is on hot encoded since each film could have one or more genre. Array size of the one hot encoding is 18 
## structure : movieid:genredata for no  word2vec embedding
## structure : movie_name_embedding:genredata for no  word2vec embedding

### In MovieEncodingEmbedding(), case-sensitivity is removed and also all characters which are not alpha numeric are removed.
### If the word isn't part of glove vocab, a 0 vector is used in its stead.
def MovieEncodingNoEmbedding(movieid):
    movieDataIDRow = movieData.loc[movieData["MovieID"]==movieid]
    movieDataMovieID = movieNameDict[movieDataIDRow.Name.values[0]]
    movieDataGenreEncoding = [1 if i in movieDataIDRow.Genres.values[0].split("|") else 0  for i in genreList]
    data = [movieDataMovieID]+movieDataGenreEncoding
    return np.array(data)

def MovieEncodingEmbedding(movieid, movieData, genreList, glove_model, vocab):
    movieDataIDRow = movieData.loc[movieData["MovieID"]==movieid]
#     print("Movie name", movieDataIDRow.Name)
    movieDataMovieID = ''.join([char for char in movieDataIDRow.Name.values[0].lower() if (char.isalpha() or char==" ")]).split(" ")
    movieNameEmbedding = np.mean([glove_model[word] if (word in vocab) else np.zeros((1,300)) for word in movieDataMovieID],axis=0)                            
    movieDataGenreEncoding = np.array([1 if i in movieDataIDRow.Genres.values[0].split("|") else 0  for i in genreList])
    data = np.concatenate((movieNameEmbedding, movieDataGenreEncoding.reshape(1,movieDataGenreEncoding.shape[0])), axis=None)
    return data.reshape(1,data.shape[0])


## User Encoding creation
## structure : gender,userDataAge,userDataOccupation,userDataZipCode 

#Gender Dict for user data
userGenderDict = {"F":0,"M":1}

def UserEncoding(userid):
    
    userDataIDRow = userData.loc[userData["UserID"]==userid]
    userDataGender = userGenderDict[userDataIDRow.Gender.values[0]]
    userDataAge = userDataIDRow.Age.values[0]
    userDataOccupation = userDataIDRow.Occupation.values[0]
    userDataZipCode = userDataIDRow.Zipcode.values[0]
    data = np.array([userDataGender,userDataAge,userDataOccupation,userDataZipCode])
    return data.reshape(1,data.shape[0])

## Function to get the movie sequence for users.
## Function currently calculates the no. of movies to consider based on the sequence length of the 90th percentile.
##can ignore nextK for now, give -1
## nextK can be used to create base sequence + extra length sequences like sequences of length 401,402... k ones for base 400.
def UserSequences(ratingData,nextK):

    timeSortedRatingData = ratingData.sort_values(['Timestamp'],ascending=True)
    timeSortedRatingDataPerUserObj = timeSortedRatingData.groupby("UserID")
    ratingDataByLength = [(data.shape[0]) for ix,data in timeSortedRatingDataPerUserObj] 
    basicRequiredLength = int(np.percentile(ratingDataByLength,10))
    if nextK != -1:
        basicRequiredLengthToTrain = int(basicRequiredLength)+NextK
    else:
        basicRequiredLengthToTrain = int(basicRequiredLength)+1
    userSequences = []
    print(basicRequiredLength)
    for idx,data in timeSortedRatingDataPerUserObj:
            userDfsize = data.shape[0]

            if userDfsize >= basicRequiredLengthToTrain:
                data.reset_index(inplace=True,drop=True)
#                 userDfBaseList = list(data[:basicRequiredLength][['UserID','MovieID','Rating']].to_records(index=False))
                userDfBaseList = list(data[:basicRequiredLength][['MovieID']].to_records(index=False))
                userSequences.append(userDfBaseList)
                '''
                ToPredictDataAll = list(data.iloc[basicRequiredLength:][['UserID','MovieID','Rating']].to_records(index=False))
                if nextK != -1:
                    userDfBaseSequences =  [userDfBaseList+ToPredictDataAll[:i+1] for i in range(NextK)]
                else:
                    userDfBaseSequences = [userDfBaseList+ToPredictDataAll[:i+1] for i in range(len(ToPredictDataAll))] 
                userSequences.append(userDfBaseSequences)
                '''
                
                
    return userSequences

def GetAllUserSequences(data, seq_length):
    timeSortedData = data.sort_values(['Timestamp'],ascending=True)
    timeSortedDataPerUserObj = timeSortedData.groupby("UserID")
    userSequences = []
    for idx,data in timeSortedDataPerUserObj:
            userDfsize = data.shape[0]

            for start in np.arange(0, userDfsize, seq_length):
                if userDfsize-start >= seq_length:
                    data.reset_index(inplace=True,drop=True)
                    userSequences.append(list(data[start:start+seq_length][['MovieID']].to_records(index=False)))
                
    return userSequences

## functions to combine user-movie-rating encoding
## embeddingFlag - whether to use embedding or no
## UserMoviePairEncoding input data - 3 element tuple of (userId,movieId,rating) and embedding flag
## UserMoviePairEncodingSequence input data - list of 3 element tuples, and embedding flag 
## use above function if you want to make for a single user and their films.
## UserMoviePairEncodingBatch input data - list of list of 3 element tuples and embedding flag, (for all users)

def UserMoviePairEncoding(data,embeddingFlag):
    
    userEncoding = UserEncoding(data[0])
    ratingEncoding = np.array(data[2]).reshape(1,1)
    if embeddingFlag:
        movieEncoding = MovieEncodingEmbedding(data[1])
        return np.concatenate((userEncoding,movieEncoding,ratingEncoding),axis=None)
    else:
        movieEncoding = MovieEncodingNoEmbedding(data[1])
        return np.concatenate((np.array([data[0]]).reshape((1,1)),userEncoding,movieEncoding,ratingEncoding),axis=None)
    
def UserMoviePairEncodingSequence(data,embeddingFlag):
    userIdDict = {}
    movieIdDict = {}
    encodingData = np.zeros((1,1000))
    total_data = []
    for i in range(len(data)):
            userId = data[i][0]
            uData = np.zeros((1,4))
            if userId in userIdDict.keys():
                uData = userIdDict[userId]
            else:
                uData = UserEncoding(userId)
                userIdDict[userId] = uData
            
            movieId = data[i][1]
            if movieId in movieIdDict.keys():
                movieData = movieIdDict[movieId]
                encodingData = np.concatenate((uData,movieData),axis=None)
            else:
                if embeddingFlag:
                    movieData = MovieEncodingEmbedding(movieId)
                    movieIdDict[movieId] = movieData
                    encodingData = np.concatenate((uData,movieData),axis=None)
                else:    
                    movieData = MovieEncodingNoEmbedding(movieId)
                    movieIdDict[movieId] = movieData
                    encodingData = np.concatenate((uData,movieData),axis=None)
            ratingData = np.array(data[i][2]).reshape(1,1)        
            encodingData =  np.concatenate((np.array([data[i][0]]).reshape((1,1)),encodingData,ratingData),axis=None)
            total_data.append(encodingData)
    
    return np.array(total_data)

def UserMoviePairEncodingBatch(data,embeddingFlag):
    userIdDict = {}
    movieIdDict = {}
    encodingData = np.zeros((1,1000))
    total_data = []
    for i in range(len(data)):
        userData = []
        for j in range(len(data[i])):
            userId = data[i][j][0]
            uData = np.zeros((1,4))
            if userId in userIdDict.keys():
                uData = userIdDict[userId]
            else:
                uData = UserEncoding(userId)
                userIdDict[userId] = uData
            
            movieId = data[i][j][1]
            if movieId in movieIdDict.keys():
                movieData = movieIdDict[movieId]
                encodingData = np.concatenate((uData,movieData),axis=None)
            else:
                if embeddingFlag:
                    movieData = MovieEncodingEmbedding(movieId)
                    movieIdDict[movieId] = movieData
                    encodingData = np.concatenate((uData,movieData),axis=None)
                else:    
                    movieData = MovieEncodingNoEmbedding(movieId)
                    movieIdDict[movieId] = movieData
                    encodingData = np.concatenate((uData,movieData),axis=None)
            ratingData = np.array(data[i][j][2]).reshape(1,1)        
            encodingData =  np.concatenate((np.array([data[i][j][0]]).reshape((1,1)),encodingData,ratingData),axis=None)
            userData.append(encodingData)
        total_data.append(np.array(userData))    
    
    return total_data