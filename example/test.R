library(bigram2vec)
library(dplyr)
library(Matrix)
Bi <- matrix(sample.int(10,200, replace = TRUE), ncol=2) %>%
  data.frame() %>%
  group_by(X1,X2) %>%
  tally()

V <- matrix(rgamma(10*2,1), 10, 2)

Y <- matrix(rpois(100,V%*%t(V)),10,10)
Y <- as(Y, "TsparseMatrix")
Y[upper.tri(Y, diag = TRUE)] <- 0L



out <- doVB_pois(Y@x, cbind(Y@i, Y@j),
          Dim = 10L,
          L = 2,
          iter=20,
          a=1.1, b=1.1,
          display_progress = TRUE)
plot(out$logprob, type = "l")
V <-out$shape/out$rate

X <- t(combn(10,2))
plot(Y[lower.tri(Y)], rowSums(V[X[,1],]*V[X[,2],]))
abline(0,1)
