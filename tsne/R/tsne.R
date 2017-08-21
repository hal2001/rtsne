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
#' signature \code{epoch_callback(ydata)} where \code{ydata} is the output
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
#' @param epsilon Learning rate value.
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
                 mom_switch_iter = 250, epsilon = 500, min_gain = 0.01,
                 initial_P_gain = 4, exaggeration_off_iter = 100) {
  if (class(X) == "dist") {
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
    ydata <- initial_config
    initial_P_gain <- 1
  } else if (init_from_PCA) {
    ydata <- .scores_matrix(X, ncol = k, verbose = TRUE)
  } else {
    ydata <- matrix(stats::rnorm(k * n), n)
  }

  P <- .x2p(X, perplexity, 1e-05)$P
  P <- 0.5 * (P + t(P))

  P[P < eps] <- eps
  P <- P / sum(P)


  P <- P * initial_P_gain
  grads <- matrix(0, nrow(ydata), ncol(ydata))
  incs <- matrix(0, nrow(ydata), ncol(ydata))
  gains <- matrix(1, nrow(ydata), ncol(ydata))
  Q <- matrix(0, nrow(P), ncol(P))

  for (iter in 1:max_iter) {
    # Don't do epoch on iteration 1, Q hasn't been calculated yet
    if ((iter %% epoch == 0 || iter == 2)  && iter != 1) {
      # epoch
      cost <- sum(apply(P * log((P + eps) / (Q + eps)), 1, sum))
      if (iter == 2) {
        message("Initial configuration, error is: ", cost)
      }
      else {
        message("Epoch: Iteration #", iter, " error is: ", cost)
      }

      if (cost < min_cost) {
        break
      }
      if (!is.null(epoch_callback)) {
        epoch_callback(ydata)
      }
    }

    sum_ydata <- apply(ydata ^ 2, 1, sum)
    num <- 1 /
      (1 + sum_ydata + sweep(-2 * ydata %*% t(ydata), 2, -t(sum_ydata)))
    diag(num) <- 0
    Q <- num / sum(num)
    if (any(is.nan(num))) {
      message("NaN in grad. descent")
    }
    Q[Q < eps] <- eps
    stiffnesses <- 4 * (P - Q) * num
    for (i in 1:n) {
      grads[i, ] <- apply(sweep(-ydata, 2, -ydata[i, ]) *
                          stiffnesses[, i], 2, sum)
    }

    gains <- (gains + 0.2) * abs(sign(grads) != sign(incs)) +
             (gains * 0.8) * abs(sign(grads) == sign(incs))
    gains[gains < min_gain] <- min_gain

    incs <- momentum * incs - epsilon * (gains * grads)

    ydata <- ydata + incs
    ydata <- sweep(ydata, 2, apply(ydata, 2, mean))

    if (iter == mom_switch_iter) {
      momentum <- final_momentum
      message("Switching to final momentum ",
              formatC(final_momentum), " at iter ", iter)
    }

    if (iter == exaggeration_off_iter && is.null(initial_config)) {
      P <- P / initial_P_gain
      message("Switching off exaggeration at iter ", iter)
    }
  }

  cost <- sum(apply(P * log((P + eps) / (Q + eps)), 1, sum))
  message("Final configuration error is: ", cost)
  if (!is.null(epoch_callback)) {
    epoch_callback(ydata)
  }

  ydata
}
