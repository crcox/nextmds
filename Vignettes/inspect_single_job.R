## inspect single job
train_df <- read.csv("manchester_refit/31/responses_Random_TRAIN.csv",header=FALSE)
head(train_df)
train <- as.matrix(train_df[,1:3]) + 1 # translate to 1-based indexes

test_df <- read.csv("manchester_refit/31/responses_Test_TEST.csv",header=FALSE)
head(test_df)
test <- as.matrix(test_df[,1:3]) + 1 # translate to 1-based indexes

model_df <- read.csv("manchester_refit/31/model.csv",header=FALSE)
head(model_df)
model <- as.matrix(model_df)

# Current "problem": error on the training set appears to be *higher* than error
# on the test set. This is counterintuitive because the model is explicitly
# being fit to the training data and only generalizing to the test set.
#
# This issue is replicated post-hoc (model_error() function is defined at the bottom of this script).
model_error(model, train)
model_error(model, test)

x <- rep(0, 10000)
for (i in 1:10000) {
  n <- nrow(test)
  m <- nrow(train)
  ix <- sample(m,n)
  x[i] <- model_error(model, train[ix,])
}

png("training_error_size_p04_1D.png", width=600, height=600)
hist(x, main="Histogram of training error on random samples of 412 responses", xlab="error")
abline(v=0.2985437, col="blue", lwd=3) # test
abline(v=0.3423165, col="red",  lwd=3) # train
legend("topright", legend=c("train","test"), lty=1, col=c("red","blue"), lwd=3)
dev.off()
model_error <- function(model, responses) {
  # Each row of responses is a (primary, alternate, target) tuple
  # Primary is the item selected
  # Alternate is the item not selected
  # Target is the reference item
  #
  # The model will be deemed correct when the distance between primary and
  # target items is less than the distance between alternate and target items.
  #
  # IMPORTANT
  # =========
  # For the sake of this function, it is easier if target is in the first
  # position, so the columns of the response matrix are re-ordered. This is
  # easier because the first column of the distance matrix among model
  # representations will provide the information needed to determine the model
  # accuracy.
  responses <- responses[,c(3,1,2)]

  error <- 0
  n <- nrow(responses)
  dim <- ncol(model)
  for (i in 1:n) {
    r <- responses[i,]
    if (dim == 1) {
      m <- matrix(model[r,], nrow=3)
    } else {
      m <- model[r,]
    }
    d <- dist(m)
    if ( d[1] >= d[2] ) {
      error <- error + 1
    }
  }
  return(error/n)
}