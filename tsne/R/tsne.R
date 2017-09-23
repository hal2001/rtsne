#' Embed a dataset using t-distributed stochastic neighbor embedding.
#'
#' @param X Input coordinates or distance matrix.
#' @param initial_config Initial coordinates for the output coordinates.
#' @param k Number of output dimensions for the embeddding.
#' @param whiten_dims Number of dimensions to use if the data is preprocessed
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
#' @param exaggerate Numerical value to multiply input probabilities by,
#' during the early exaggeration phase. Not used if \code{initial_config} is not
#' \code{NULL}. May also provide the string \code{"ls"}, in which case the
#' dataset-dependent exaggeration technique suggested by Linderman and
#' Steinerberger (2017) is used.
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
#' @references
#' Van der Maaten, L., & Hinton, G. (2008).
#' Visualizing data using t-SNE.
#' \emph{Journal of Machine Learning Research}, \emph{9} (2579-2605).
#' \url{http://www.jmlr.org/papers/v9/vandermaaten08a.html}
#'
#' Linderman, G. C., & Steinerberger, S. (2017).
#' Clustering with t-SNE, provably.
#' \emph{arXiv preprint} \emph{arXiv}:1706.02582.
#' \url{https://arxiv.org/abs/1706.02582}
#' @export
tsne <- function(X, initial_config = NULL, k = 2,
                 perplexity = 30, max_iter = 1000,
                 whiten = FALSE, whiten_dims = 30,
                 init_from_PCA = FALSE,
                 epoch_callback = NULL, epoch = 100, min_cost = 0,
                 momentum = 0.5, final_momentum = 0.8, mom_switch_iter = 250,
                 eta = 500, min_gain = 0.01,
                 exaggerate = 4, exaggeration_off_iter = 100) {

  if (methods::is(X, "dist")) {
    n <- attr(X, "Size")
  } else {
    message("Range scaling X")
    X <- as.matrix(X)
    X <- X - min(X)
    X <- X / max(X)
    whiten_dims <- min(whiten_dims, ncol(X))
    if (whiten) {
      message("Whitening")
      X <- .whiten(as.matrix(X), n.comp = whiten_dims)
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
    exaggerate <- 1
  } else {
    if (init_from_PCA) {
      Y <- .scores_matrix(X, ncol = k, verbose = TRUE)
    } else {
      Y <- matrix(stats::rnorm(k * n), n)
    }
  }

  P <- .x2p(X, perplexity, 1e-05)$P
  P <- 0.5 * (P + t(P))

  P[P < eps] <- eps
  P <- P / sum(P)

  # Used during Linderman-Steinerberger exaggeration
  ls_exaggerate <- 0.1 * n
  if (tolower(exaggerate) == "ls") {
    message("Linderman-Steinerberger exaggeration = ", formatC(ls_exaggerate))
    P <- P * ls_exaggerate
  }
  else {
    P <- P * exaggerate
  }

  grads <- matrix(0, n, k)
  incs <- matrix(0, n, k)
  gains <- matrix(1, n, k)
  Q <- matrix(0, n, n)

  for (iter in 1:max_iter) {
    D2 <- rowSums(Y * Y)
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

    if (tolower(exaggerate) == "ls" && iter <= exaggeration_off_iter) {
      # during LS exaggeration, use gradient descent only with eta = 1
      incs <- -grads
    }
    else {
      gains <- (gains + 0.2) * abs(sign(grads) != sign(incs)) +
        (gains * 0.8) * abs(sign(grads) == sign(incs))
      gains[gains < min_gain] <- min_gain
      incs <- momentum * incs - eta * (gains * grads)
    }

    Y <- Y + incs
    Y <- sweep(Y, 2, colMeans(Y))

    if (iter == mom_switch_iter) {
      momentum <- final_momentum
      message("Switching to final momentum ",
              formatC(final_momentum), " at iter ", iter)
    }

    if (iter == exaggeration_off_iter && is.null(initial_config)) {
      if (tolower(exaggerate) == "ls") {
        message("Switching off Linderman-Steinerberger exaggeration at iter ",
                iter)
        P <- P / ls_exaggerate
      } else {
        message("Switching off exaggeration at iter ", iter)
        P <- P / exaggerate
      }
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
