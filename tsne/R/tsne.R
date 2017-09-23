#' Embed a dataset using t-distributed stochastic neighbor embedding.
#'
#' @param X Input coordinates or distance matrix.
#' @param initial_config Initial coordinates for the output coordinates.
#' @param k Number of output dimensions for the embeddding.
#' @param initial_dims Number of dimensions to use if the data is preprocessed
#' by whitening. Must not be greater than the number of columns in \code{X}.
#' @param perplexity The target perplexity for parameterizing the input
#' probabilities.
#' @param max_iter Maximum number of iterations in the optimization.
#' @param min_cost If the cost falls below this value, the optimization will
#' stop early.
#' @param epoch_callback Function to call after each epoch. Should have the
#' signature \code{epoch_callback(Y)} where \code{Y} is the output
#' coordinate matrix.
#' @param whiten If \code{TRUE}, whitens the input data before calculating the
#' input probabilities.
#' @param init_from_PCA if set to \code{TRUE}, output coordinates are
#' initialized from the first \code{k} scores of the PCA. If not \code{TRUE},
#' then the output coordinates are initialized from a Gaussian distribution
#' with mean 0 and standard deviation 1e-4. Ignored if \code{initial_config} is
#' not \code{NULL}.
#' @param epoch After every \code{epoch} number of steps, calculates and
#' displays the cost value and calls \code{epoch_callback}, if supplied.
#' @param momentum Initial momentum value.
#' @param final_momentum Final momentum value.
#' @param mom_switch_iter Iteration at which the momentum will switch from
#' \code{momentum} to \code{final_momentum}.
#' @param eta Learning rate value.
#' @param min_gain Minimum gradient descent step size.
#' @param initial_P_gain Value to multiply input probabilities by, during the
#' early exaggeration phase. Not used if \code{initial_config} is not
#' \code{NULL}.
#' @param exaggeration_off_iter Iteration at which early exaggeration is turned
#' off.
#' @return The embedded output coordinates.
#' @examples
#' \dontrun{
#' colors = rainbow(length(unique(iris$Species)))
#' names(colors) = unique(iris$Species)
#' ecb = function(x, y) {
#'   plot(x, t = 'n')
#'   text(x, labels = iris$Species, col = colors[iris$Species])
#' }
#' tsne_iris = tsne(iris[, 1:4], epoch_callback = ecb, perplexity = 50)
#' }
#' @export
tsne <- function(X, initial_config = NULL, k = 2, initial_dims = 30,
                 perplexity = 30, max_iter = 1000, min_cost = 0,
                 epoch_callback = NULL, whiten = TRUE, init_from_PCA = FALSE,
                 epoch = 100, momentum = 0.5, final_momentum = 0.8,
                 mom_switch_iter = 250, eta = 500, min_gain = 0.01,
                 initial_P_gain = 4, exaggeration_off_iter = 100) {
  if (methods::is(X, "dist")) {
    n <- attr(X, "Size")
  } else {
    message("Range scaling X")
    X <- as.matrix(X)
    X <- X - min(X)
    X <- X / max(X)
    initial_dims <- min(initial_dims, ncol(X))
    if (whiten) {
      message("Whitening")
      X <- .whiten(as.matrix(X), n.comp = initial_dims)
    }
    n <- nrow(X)
  }

  eps <- .Machine$double.eps # machine precision

  if (!is.null(initial_config) && is.matrix(initial_config)) {
    if (nrow(initial_config) != n | ncol(initial_config) != k) {
      stop(
        "initial_config argument does not match necessary configuration for X")
    }
    Y <- initial_config
    initial_P_gain <- 1
  } else if (init_from_PCA) {
    Y <- .scores_matrix(X, ncol = k, verbose = TRUE)
  } else {
    Y <- matrix(stats::rnorm(k * n), n)
  }

  P <- .x2p(X, perplexity, 1e-05)$P
  P <- 0.5 * (P + t(P))

  P[P < eps] <- eps
  P <- P / sum(P)

  P <- P * initial_P_gain
  grads <- matrix(0, nrow(Y), ncol(Y))
  incs <- matrix(0, nrow(Y), ncol(Y))
  gains <- matrix(1, nrow(Y), ncol(Y))
  Q <- matrix(0, nrow(P), ncol(P))

  for (iter in 1:max_iter) {
    D2 <- apply(Y ^ 2, 1, sum)
    D2 <- D2 + sweep(-2 * Y %*% t(Y), 2, -t(D2))
    W <- 1 / (1 + D2)
    diag(W) <- 0
    Q <- W / sum(W)
    if (any(is.nan(W))) {
      message("NaN in grad. descent")
    }
    Q[Q < eps] <- eps
    K <- 4 * (P - Q) * W
    for (i in 1:n) {
      grads[i, ] <- colSums(sweep(-Y, 2, -Y[i, ]) * K[, i])
    }

    gains <- (gains + 0.2) * abs(sign(grads) != sign(incs)) +
             (gains * 0.8) * abs(sign(grads) == sign(incs))
    gains[gains < min_gain] <- min_gain

    incs <- momentum * incs - eta * (gains * grads)

    Y <- Y + incs
    Y <- sweep(Y, 2, colMeans(Y))

    if (iter == mom_switch_iter) {
      momentum <- final_momentum
      message("Switching to final momentum ",
              formatC(final_momentum), " at iter ", iter)
    }

    if (iter == exaggeration_off_iter && is.null(initial_config)) {
      P <- P / initial_P_gain
      message("Switching off exaggeration at iter ", iter)
    }

    if (iter %% epoch == 0) {
      # epoch
      cost <- sum(apply(P * log((P + eps) / (Q + eps)), 1, sum))
      message("Epoch: Iteration #", iter, " error is: ", cost)
      if (cost < min_cost) {
        break
      }
    }
    if (!is.null(epoch_callback)) {
      epoch_callback(Y)
    }
  }

  Y
}
