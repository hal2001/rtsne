# Given a vector of distances and an exponential parameter beta, calculates
# the probabilities and corresponding Shannon entropy.
#
# Returns a list containing the Shannon entropy and the probability.
.Hbeta <- function(D, beta) {
  P <- exp(-D * beta)
  sumP <- sum(P)
  if (sumP == 0) {
    H <- 0
    P <- D * 0
  } else {
    H <- log(sumP) + beta * sum(D * P) / sumP
    P <- P / sumP
  }
  list(H = H, P = P)
}

# Calculates the input probabilities from X, such that each row probability
# distribution has the specified perplexity (within the supplied tolerance).
# Returns a list containing the probabilities and beta values.
# NB the default kernel, "exp", differs from the procedure in the TSNE paper by
# exponentially weighting the distances, rather than the squared distances.
# Set the kernel to "gauss" to get the squared distance version.
.x2p <- function(X, perplexity = 15, tol = 1e-5, kernel = "exp",
                  verbose = FALSE) {
  x_is_dist <- methods::is(X, "dist")
  if (x_is_dist) {
    D <- X
    n <- attr(D, "Size")

    D <- as.matrix(D)
    if (kernel == "gauss") {
      D <- D * D
    }
  }
  else {
    XX <- rowSums(X * X)
    n <- nrow(X)
  }

  P <- matrix(0, n, n)
  beta <- rep(1, n)
  logU <- log(perplexity)
  perps <- rep(1, n)

  for (i in 1:n) {
    betamin <- -Inf
    betamax <- Inf

    if (x_is_dist) {
      Di <- D[i, -i]
    }
    else {
      Di <- (XX[i] + XX - 2 * as.vector(X %*% X[i, ]))[-i]
      Di[Di < 0] <- 0
      if (kernel == "exp") {
        Di <- sqrt(Di)
      }
    }
    # Initialization used for all points in ELKI according to Schubert & Gertz
    # in "Intrinsic t-Stochastic Neighbor Embedding for Visualization and
    # Outlier Detection: A Remedy Against the Curse of Dimensionality?"
    # Using the last optimized beta seems to be better most of the time based
    # on my testing though, so we'll only use it for the first point.
    if (i == 1) {
      beta[1] <- 0.5 * perplexity / mean(Di)
    }

    hbeta <- .Hbeta(Di, beta[i])
    H <- hbeta$H
    thisP <- hbeta$P
    Hdiff <- H - logU
    tries <- 0

    while (abs(Hdiff) > tol && tries < 50) {
      if (Hdiff > 0) {
        betamin <- beta[i]
        if (is.infinite(betamax)) {
          beta[i] <- beta[i] * 2
        } else {
          beta[i] <- (beta[i] + betamax) / 2
        }
      } else {
        betamax <- beta[i]
        if (is.infinite(betamin)) {
          beta[i] <- beta[i] / 2
        } else {
          beta[i] <- (beta[i] + betamin) / 2
        }
      }

      hbeta <- .Hbeta(Di, beta[i])
      H <- hbeta$H
      thisP <- hbeta$P
      Hdiff <- H - logU
      tries <- tries + 1
    }
    # initialize guess for next point with optimized beta for this point
    # doesn't save many iterations, but why not?
    if (i < n) {
      beta[i + 1] <- beta[i]
    }
    P[i, -i] <- thisP
    perps[i] <- exp(H)
  }
  sigma <- sqrt(1 / beta)

  if (verbose) {
    summary_sigma <- summary(sigma, digits = max(3, getOption("digits") - 3))
    message(date(), " sigma summary: ",
            paste(names(summary_sigma), ":", summary_sigma, "|", collapse = ""))
  }
  list(P = P, beta = beta)
}

# Calculates a matrix containing the first ncol columns of the PCA scores.
# Returns the score matrix unless ret_extra is TRUE, in which case a list
# is returned also containing the eigenvalues
.scores_matrix <- function(X, ncol = min(dim(X)),
                           verbose = FALSE, ret_extra = FALSE) {
  X <- scale(X, center = TRUE, scale = FALSE)
  # do SVD on X directly rather than forming covariance matrix
  ncomp <- ncol
  s <- svd(X, nu = ncomp, nv = 0)
  D <- diag(c(s$d[1:ncomp]))
  if (verbose || ret_extra) {
    # calculate eigenvalues of covariance matrix from singular values
    lambda <- (s$d ^ 2) / (nrow(X) - 1)
    varex <- sum(lambda[1:ncomp]) / sum(lambda)
    message("PCA: ", ncomp, " components explained ", formatC(varex * 100),
            "% variance")
  }
  scores <- s$u %*% D
  if (ret_extra) {
    list(
      scores = scores,
      lambda = lambda[1:ncomp]
    )
  }
  else {
    scores
  }
}

# Whiten the data by PCA. This both reduces the dimensionality, but also
# scales the scores by the inverse square root of the equivalent eigenvalue
# so that the variance of each column is 1.
.whiten_pca <- function(X, ncol = min(dim(X)), eps = 1e-5, verbose = FALSE) {
  pca <- .scores_matrix(X, ncol = ncol, verbose = verbose, ret_extra = TRUE)
  sweep(pca$scores, 2, sqrt(pca$lambda + eps), "/")
}

