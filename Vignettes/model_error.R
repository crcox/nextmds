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
  responses <- as.matrix(responses[,c(3,1,2)])
  if (min(responses)==0) {
    responses <- responses + 1
  }
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
