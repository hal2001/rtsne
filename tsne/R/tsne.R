#' Embed a dataset using t-distributed stochastic neighbor embedding.
#'
#' @param X Input coordinates or distance matrix.
#' @param k Number of output dimensions for the embedding.
#' @param scale How to preprocess \code{X}. One of: \code{"none"} or
#'   (\code{NULL}), which applies no further preprocessing; \code{"range"},
#'   which range scales the matrix elements between 0 and 1; \code{"bh"}, which
#'   applies the same scaling in Barnes-Hut t-SNE, where the columns are mean
#'   centered and then the elements divided by absolute maximum value.
#' @param init How to initialize the output coordinates. One of: \code{"rand"},
#'   which initializes from a Gaussian distribution with mean 0 and standard
#'   deviation 1e-4; \code{"pca"}, which uses the first \code{k} scores of the
#'   PCA: columns are centered, but no scaling beyond that which is applied by
#'   the \code{scale} parameter is carried out; \code{"spca"}, which uses the
#'   PCA scores and then scales each score to a standard deviation of 1e-4; or a
#'   matrix can be used to set the coordinates directly. It must have dimensions
#'   \code{n} by \code{k}, where \code{n} is the number of rows in \code{X}.
#' @param perplexity The target perplexity for parameterizing the input
#'   probabilities.
#' @param inp_kernel The input kernel function. Can be either \code{"gauss"}
#'   (the default), or \code{"exp"}, which uses the unsquared distances.
#'   \code{"exp"} is not the usual literature function, but matches the original
#'   rtsne implementation (and it probably doesn't matter very much).
#' @param max_iter Maximum number of iterations in the optimization.
#' @param whiten If \code{TRUE}, whitens the input data before calculating the
#'   input probabilities.
#' @param whiten_dims Number of dimensions to use if the data is preprocessed by
#'   whitening. Must not be greater than the number of columns in \code{X}.
#' @param min_cost If the cost falls below this value, the optimization will
#'   stop early.
#' @param epoch_callback Function to call after each epoch. Should have the
#'   signature \code{epoch_callback(Y)} where \code{Y} is the output coordinate
#'   matrix.
#' @param epoch After every \code{epoch} number of steps, calculates and
#'   displays the cost value and calls \code{epoch_callback}, if supplied.
#' @param momentum Initial momentum value.
#' @param final_momentum Final momentum value.
#' @param mom_switch_iter Iteration at which the momentum will switch from
#'   \code{momentum} to \code{final_momentum}.
#' @param eta Learning rate value.
#' @param min_gain Minimum gradient descent step size.
#' @param exaggerate Numerical value to multiply input probabilities by, during
#'   the early exaggeration phase. Not used if \code{initial_config} is not
#'   \code{NULL}. May also provide the string \code{"ls"}, in which case the
#'   dataset-dependent exaggeration technique suggested by Linderman and
#'   Steinerberger (2017) is used.
#' @param exaggeration_off_iter Iteration at which early exaggeration is turned
#'   off.
#' @param verbose If \code{TRUE}, log progress messages to the console.
#' @return The embedded output coordinates.
#' @examples
#' \dontrun{
#' colors = rainbow(length(unique(iris$Species)))
#' names(colors) = unique(iris$Species)
#' ecb = function(x, y) {
#'   plot(x, t = 'n')
#'   text(x, labels = iris$Species, col = colors[iris$Species])
#' }
#' # verbose = TRUE logs progress to console
#' tsne_iris <- tsne(iris[, -5], epoch_callback = ecb, perplexity = 50, verbose = TRUE)
#' # Use the early exaggeration suggested by Linderman and Steinerberger
#' tsne_iris_ls <- tsne(iris[, -5], epoch_callback = ecb, perplexity = 50, exaggerate = "ls")
#' # Make embedding deterministic by initializing with scaled PCA scores
#' tsne_iris_spca <- tsne(iris[, -5], epoch_callback = ecb, perplexity = 50, exaggerate = "ls",
#'                        scale = "spca")
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
tsne <- function(X, k = 2, scale = "range", init = "rand",
                 perplexity = 30, inp_kernel = "gauss", max_iter = 1000,
                 whiten = FALSE, whiten_dims = 30,
                 epoch_callback = NULL, epoch = 100, min_cost = 0,
                 momentum = 0.5, final_momentum = 0.8, mom_switch_iter = 250,
                 eta = 500, min_gain = 0.01,
                 exaggerate = 4, exaggeration_off_iter = 100,
                 verbose = FALSE) {

  if (methods::is(X, "dist")) {
    n <- attr(X, "Size")
  } else {
    if (!is.null(scale)) {
      scale <- match.arg(tolower(scale), c("none", "range", "bh"))

      switch(scale,
        range = {
          if (verbose) {
            message(date, " Range scaling X")
          }
          X <- as.matrix(X)
          X <- X - min(X)
          X <- X / max(X)
        },
        bh = {
          if (verbose) {
            message(date(), " Normalizing BH-style")
          }
          X <- base::scale(X, scale = FALSE)
          X <- X / abs(max(X))
        }
      )
    }

    whiten_dims <- min(whiten_dims, ncol(X))
    if (whiten) {
      if (verbose) {
        message(date(), " Whitening")
      }
      X <- .whiten(as.matrix(X), n.comp = whiten_dims)
    }
    n <- nrow(X)
  }

  if (!is.null(init)) {
    if (methods::is(init, "matrix")) {
      if (nrow(init) != n || ncol(init) != k) {
        stop("init matrix does not match necessary configuration for X")
      }
      Y <- init
      exaggerate <- 1
    }
    else {
      init <- match.arg(tolower(init), c("rand", "pca", "spca"))
      Y <- switch(init,
        pca = {
          if (verbose) {
            message(date(), " Initializing from PCA scores")
          }
          .scores_matrix(X, ncol = k, verbose = verbose)
        },
        spca = {
          if (verbose) {
            message(date(), " Initializing from scaled PCA scores")
          }
          scores <- .scores_matrix(X, ncol = k, verbose = verbose)
          scale(scores, scale = apply(scores, 2, stats::sd) / 1e-4)
        },
        rand = {
          if (verbose) {
            message(date(), " Initializing from random Gaussian with sd = 1e-4")
          }
          matrix(stats::rnorm(k * n, sd = 1e-4), n)
        }
      )
    }
  }

  # Display initialization
  if (!is.null(epoch_callback)) {
    epoch_callback(Y)
  }

  eps <- .Machine$double.eps # machine precision

  P <- .x2p(X, perplexity, 1e-5, kernel = inp_kernel, verbose = verbose)$P
  P <- 0.5 * (P + t(P))
  P[P < eps] <- eps
  P <- P / sum(P)

  # Used during Linderman-Steinerberger exaggeration
  ls_exaggerate <- 0.1 * n
  if (tolower(exaggerate) == "ls") {
    if (verbose) {
      message("Linderman-Steinerberger exaggeration = ", formatC(ls_exaggerate))
    }
    P <- P * ls_exaggerate
  }
  else {
    P <- P * exaggerate
  }

  G <- matrix(0, n, k)
  uY <- matrix(0, n, k)
  gains <- matrix(1, n, k)
  Q <- matrix(0, n, n)

  if (max_iter < 1) {
    return(Y)
  }

  for (iter in 1:max_iter) {
    # D2
    Q <- rowSums(Y * Y)
    Q <- Q + sweep(-2 * Y %*% t(Y), 2, -t(Q))
    # W
    Q <- 1 / (1 + Q)
    diag(Q) <- 0
    if (any(is.nan(Q))) {
      message("NaN in grad. descent")
      # Give up and return the last iteration's result
      return(Y)
    }
    sumW <- sum(Q)
    # Q
    Q <- Q / sum(Q)
    Q[Q < eps] <- eps

    K <- 4 * (P - Q) * Q * sumW
    for (i in 1:n) {
      G[i, ] <- colSums(sweep(-Y, 2, -Y[i, ]) * K[, i])
    }

    if (tolower(exaggerate) == "ls" && iter <= exaggeration_off_iter) {
      # during LS exaggeration, use gradient descent only with eta = 1
      uY <- -G
    }
    else {
      # compare signs of G with -update (== previous G, if we ignore momentum)
      # abs converts TRUE/FALSE to 1/0
      dbd <- abs(sign(G) != sign(uY))
      gains <- (gains + 0.2) * dbd + (gains * 0.8) * (1 - dbd)
      gains[gains < min_gain] <- min_gain
      uY <- momentum * uY - eta * gains * G
    }

    Y <- Y + uY
    # Recenter Y
    Y <- sweep(Y, 2, colMeans(Y))

    if (iter == mom_switch_iter) {
      momentum <- final_momentum
      if (verbose) {
        message("Switching to final momentum ", formatC(final_momentum),
                " at iter ", iter)
      }
    }

    if (iter == exaggeration_off_iter && !methods::is(init, "matrix")) {
      if (tolower(exaggerate) == "ls") {
        if (verbose) {
          message("Switching off Linderman-Steinerberger exaggeration at iter ",
                  iter)
        }
        P <- P / ls_exaggerate
      } else {
        if (verbose) {
          message("Switching off exaggeration at iter ", iter)
        }
        P <- P / exaggerate
      }
    }

    if (iter %% epoch == 0) {
      # epoch
      cost <- sum(P * log((P + eps) / (Q + eps)))
      if (verbose) {
        message(date(), " Epoch: Iteration #", iter, " error is: ",
                formatC(cost))
      }
      if (cost < min_cost) {
        break
      }
      if (!is.null(epoch_callback)) {
        epoch_callback(Y)
      }
    }
  }

  Y
}
