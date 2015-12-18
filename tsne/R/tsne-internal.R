# Given a vector of distances and an exponential parameter beta, calculates
# the probabilities and corresponding Shannon entropy.
#
# NB this differs from the procedure in the TSNE paper by exponentially
# weighting the distances, rather than the squared distances.
#
# Returns a list containing the Shannon entropy and the probability.
.Hbeta <- function(D, beta) {
  P <- exp(-D * beta)
  sumP <- sum(P)
  if (sumP == 0) {
    H <- 0
    P <- D * 0
  } else {
    H <- log(sumP) + beta * sum(D %*% P) / sumP
    P <- P / sumP
  }
  list(H = H, P = P)
}

# Calculates the input probabilities from X, such that each row probability
# distribution has the specified perplexity (within the supplied tolerance).
# Returns a list containing the probabilities and beta values.
.x2p <- function(X, perplexity = 15, tol = 1e-05) {
  if (class(X) == "dist") {
      D <- X
  } else {
      D <- dist(X)
  }
  n <- attr(D, "Size")

  D <- as.matrix(D)
  P <- matrix(0, n, n)
  beta <- rep(1, n)
  logU <- log(perplexity)
  perps <- rep(1, n)

  for (i in 1:n) {
    betamin <- -Inf
    betamax <- Inf
    Di <- D[i, -i]
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
    P[i, -i] <- thisP
    perps[i] <- exp(H)
  }
  sigma = sqrt(1 / beta)

  r  <- list(P = P, beta = beta)

  message("sigma summary: ", paste(names(summary(sigma)), ":", summary(sigma),
                                   "|", collapse = ""))
  message("beta summary: ", paste(names(summary(beta)), ":", summary(beta),
                                  "|", collapse = ""))
  message("perp summary: ", paste(names(summary(perps)), ":", summary(perps),
                                  "|", collapse = ""))

  r
}

# Whitens the input matrix X using n.comp components.
# Returns the whitened matrix.
.whiten <- function(X, row.norm = FALSE, verbose = FALSE, n.comp = ncol(X)) {
  n.comp  # forces an eval/save of n.comp
  if (verbose) {
    message("Centering")
  }
  n <- nrow(X)
  p <- ncol(X)
  X <- scale(X, scale = FALSE)
  if (row.norm) {
    X <- t(scale(X, scale = row.norm))
  } else {
    t(X)
  }

  if (verbose) {
    message("Whitening")
  }
  V <- X %*% t(X) / n
  s <- La.svd(V)
  D <- diag(c(1 / sqrt(s$d)))
  K <- D %*% t(s$u)
  K <- matrix(K[1:n.comp, ], n.comp, p)
  X <- t(K %*% X)
  X
}

# Calculates a matrix containing the first ncol columns of the PCA scores.
# Returns the score matrix.
.scores_matrix <- function(X, ncol = min(nrow(X), base::ncol(X)),
                           verbose = FALSE) {
  X <- scale(X, center = TRUE, scale = FALSE)
  # do SVD on X directly rather than forming covariance matrix
  ncomp <- ncol
  s <- svd(X, nu = ncomp, nv = 0)
  D <- diag(c(s$d[1:ncomp]))
  if (verbose) {
    # calculate eigenvalues of covariance matrix from singular values
    lambda <- (s$d ^ 2) / (nrow(X) - 1)
    varex <- sum(lambda[1:ncomp]) / sum(lambda)
    message("PCA: ", ncomp, " components explained ", formatC(varex * 100),
            "% variance")
  }
  s$u %*% D
}
