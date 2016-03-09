## inspect single job
source('./model_error.R')
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
