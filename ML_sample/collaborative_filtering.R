library(recommenderlab)	 	 

## Loading rating data
rating.data <- read.csv("rating.csv")
rating.data

# preprocess a little:
rownames(rating.data) <- rating.data[,1] # first column contains users names
rating.data <- rating.data[,-1] # remove first column form rating matrix 
rating.data

# convert data.frame to matrix
rating.matrix <- as.matrix(rating.data)

# convert rating matrix to real rating matrix
# (real rating matrix is a matrix format that 
# recommenredlab uses to compute recommendations)
real.rating.matrix<- as(rating.matrix,"realRatingMatrix")

# inspect the data
dimnames(rating.matrix)
rowMeans(rating.matrix)


## recommendation systems:
# let us start with User Based Collaborative Filtering (UBCF)
# Option1:  UBCF, trained on the first 999 users (out of 1000)
UB.Rec1 <- Recommender(real.rating.matrix[1:999], method = "UBCF")
UB.Rec1
# Option2: UBCF, trained on 999 users, with normalized training data
UB.Rec2 <- Recommender(real.rating.matrix[1:999], method = "UBCF", 
                      param=list(normalize = "Z-score", method = "Cosine"))
UB.Rec2


## Now let us predict the rating for user u1000 (using the second model UB.Rec2, 
# you are welcome to test the other model too)
# Note!! rating is provided only for items that the user have not yet rated
predicted.rating.u10 <- predict(UB.Rec2, real.rating.matrix[1000], type = "ratings")
# inspest the prediction: 
as(predicted.rating.u10, "matrix")


# Now, let us generate an Item Based Collaborative Filterng - based recommendation
# all we need to do is to replace the method
# NOTE: the method generates recommendation per user - the ouput is in 
# the same format as UBCF
# The only difference is that the recommendation is based on different network data:
# UBCF: network between users, based on similarity between their rating
# IBCF: network between users, based on similarity between the *items* they have rates
IB.Rec1 <- Recommender(real.rating.matrix[1:999], method = "IBCF")
IB.Rec1

# similarly, we can now predict the rating of user 1000
predicted.rating.u10 <- predict(IB.Rec1, real.rating.matrix[1000], type = "ratings")
# inspest the prediction: 
as(predicted.rating.u10, "matrix")


### Evaluating the output (not part of class material, but worth knowing)
# let us evaluate the rating that IBCT and UBCF generate. 
# For that, we will split the data to: 
# 60% of the users are used for training
# for the reminder 40% users: 
# 30 items are used for prediction
# all other items are user for evaluation
split.data <- evaluationScheme(real.rating.matrix[1:1000], 
                               method = "split", train = 0.6, given = 30)
# UBCF
UB.Rec <- Recommender(getData(split.data, "train"), "UBCF")
# IBCF
IB.Rec <- Recommender(getData(split.data, "train"), "IBCF")

# predict rating of users in validation, based on 30 known items
pred.UB <- predict(UB.Rec, getData(split.data, "known"), type="ratings")
pred.IB <- predict(IB.Rec, getData(split.data, "known"), type="ratings")

# obtaining the error based on the reminder unknown items of users in the validation set
error.UB <- calcPredictionAccuracy(pred.UB, getData(split.data, "unknown"))
error.IB <- calcPredictionAccuracy(pred.IB, getData(split.data, "unknown"))

error.UB
error.IB
