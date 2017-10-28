#' t-Distributed Stochastic Neighbor Embedding
#'
#' Embed a dataset using t-SNE.
#'
#' @param X Input coordinates or distance matrix.
#' @param k Number of output dimensions for the embedding.
#' @param scale If \code{TRUE}, scale each column to zero mean and unit
#'   variance. Alternatively, you may specify one of the following strings:
#'   \code{"range"}, which range scales the matrix elements between 0 and 1;
#'   \code{"bh"}, which applies the same scaling in Barnes-Hut t-SNE, where the
#'   columns are mean centered and then the elements divided by absolute maximum
#'   value; \code{"scale"} does the same as using \code{TRUE}. To use the input
#'   data as-is, use \code{FALSE}, \code{NULL} or \code{"none"}.
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
#' @param exaggeration_factor Numerical value to multiply input probabilities by, during
#'   the early exaggeration phase. Not used if \code{initial_config} is not
#'   \code{NULL}. May also provide the string \code{"ls"}, in which case the
#'   dataset-dependent exaggeration technique suggested by Linderman and
#'   Steinerberger (2017) is used.
#' @param stop_lying_iter Iteration at which early exaggeration is turned
#'   off.
#' @param ret_extra If \code{TRUE}, return value is a list containing additional
#'   details on the t-SNE procedure; otherwise just the output coordinates. See
#'   the \code{Value} section for more.
#' @param verbose If \code{TRUE}, log progress messages to the console.
#' @return If \code{ret_extra} is \code{FALSE}, the embedded output coordinates
#'   as a matrix. Otherwise, a list with the following items:
#' \itemize{
#' \item{\code{Y}} Matrix containing the embedded output coordinates.
#' \item{\code{N}} Number of objects.
#' \item{\code{origD}} Dimensionality of the input data.
#' \item{\code{scale}} Scaling applied to input data, as specified by the
#'   \code{scale} parameter.
#' \item{\code{init}} Initialization type of the output coordinates, as
#'   specified by the \code{init} parameter, or if a matrix was used, this will
#'   contain the string \code{"matrix"}.
#' \item{\code{iter}} Number of iterations the optimization carried out.
#' \item{\code{time_secs}} Time taken for the embedding, in seconds.
#' \item{\code{perplexity}} Target perplexity of the input probabilities, as
#'   specified by the \code{perplexity} parameter.
#' \item{\code{costs}} Embedding error associated with each observation. This is
#'   the sum of the absolute value of each component of the KL cost that the
#'   observation is associated with, so don't expect these to sum to the
#'   reported KL cost.
#' \item{\code{itercosts}} KL cost at each epoch.
#' \item{\code{stop_lying_iter}} Iteration at which early exaggeration is
#'   stopped, as specified by the \code{stop_lying_iter} parameter.
#' \item{\code{mom_switch_iter}} Iteration at which momentum used in
#'   optimization switches from \code{momentum} to \code{final_momentum}, as
#'   specified by the \code{mom_switch_iter} parameter.
#' \item{\code{momentum}} Momentum used in the initial part of the optimization,
#'   as specified by the \code{momentum} parameter.
#' \item{\code{final_momentum}} Momentum used in the second part of the
#'   optimization, as specified by the \code{final_momentum} parameter.
#' \item{\code{eta}} Learning rate, as specified by the \code{eta} parameter.
#' \item{\code{exaggeration_factor}} Multiplier of the input probabilities
#'   during the exaggeration phase. If the Linderman-Steinerberger exaggeration
#'   scheme is used, this value will have the name \code{"ls"}.
#' }
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
#' tsne_iris_ls <- tsne(iris[, -5], epoch_callback = ecb, perplexity = 50,
#'                      exaggeration_factor = "ls")
#' # Make embedding deterministic by initializing with scaled PCA scores
#' tsne_iris_spca <- tsne(iris[, -5], epoch_callback = ecb, perplexity = 50,
#'                        exaggeration_factor = "ls", scale = "spca")
#' # Return extra details about the embedding
#' tsne_iris_extra <- tsne(iris[, -5], epoch_callback = ecb, perplexity = 50,
#'                         exaggeration_factor = "ls", scale = "spca", ret_extra = TRUE)
#'
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
                 epoch_callback = NULL, epoch = base::round(max_iter / 10),
                 min_cost = 0,
                 momentum = 0.5, final_momentum = 0.8, mom_switch_iter = 250,
                 eta = 500, min_gain = 0.01,
                 exaggeration_factor = 4, stop_lying_iter = 100,
                 ret_extra = FALSE,
                 verbose = FALSE) {

  start_time <- NULL
  if (ret_extra) {
    start_time <- Sys.time()
  }

  if (methods::is(X, "dist")) {
    n <- attr(X, "Size")
  }
  else {
    if (methods::is(X, "data.frame")) {
      indexes <- which(vapply(X, is.numeric, logical(1)))
      if (verbose) {
        message("Found ", length(indexes), " numeric columns")
      }
      if (length(indexes) == 0) {
        stop("No numeric columns found")
      }
      X <- X[, indexes]
    }

    if (is.null(scale)) {
      scale <- "none"
    }
    if (is.logical(scale)) {
      if (scale) {
        scale <- "scale"
      }
      else {
        scale <- "none"
      }
    }
    scale <- match.arg(tolower(scale), c("none", "scale", "range", "bh"))

    switch(scale,
      range = {
        if (verbose) {
          message(date(), " Range scaling X")
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
      },
      scale = {
        if (verbose) {
          message(date(), " Scaling to zero mean and unit variance")
        }
        X <- Filter(stats::var, X)
        if (verbose) {
          message("Kept ", ncol(X), " non-zero-variance columns")
        }
        X <- base::scale(X, scale = TRUE)
      },
      none = {
        X <- as.matrix(X)
      }
    )


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
      init <- "matrix"
      exaggeration_factor <- 1
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
    do_callback(epoch_callback, Y, 0)
  }

  itercosts <- c()
  if (tolower(exaggeration_factor) == "ls") {
    # Linderman-Steinerberger exaggeration
    exaggeration_factor <- 0.1 * n
    names(exaggeration_factor) <- "ls"
  }
  else {
    names(exaggeration_factor) <- "ex"
  }

  if (max_iter < 1) {
    return(ret_value(Y, ret_extra, X, scale, init, iter = 0,
                     start_time = start_time))
  }

  eps <- .Machine$double.eps # machine precision

  P <- .x2p(X, perplexity, 1e-5, kernel = inp_kernel, verbose = verbose)$P
  P <- 0.5 * (P + t(P))
  P[P < eps] <- eps
  P <- P / sum(P)

  if (names(exaggeration_factor) == "ls") {
    if (verbose) {
      message("Linderman-Steinerberger exaggeration = ", formatC(exaggeration_factor))
    }
  }
  P <- P * exaggeration_factor

  G <- matrix(0, n, k)
  uY <- matrix(0, n, k)
  gains <- matrix(1, n, k)
  Q <- matrix(0, n, n)
  mu <- momentum

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
      break
    }
    sumW <- sum(Q)
    # Q
    Q <- Q / sumW
    Q[Q < eps] <- eps
    K <- 4 * (P - Q) * Q * sumW
    for (i in 1:n) {
      G[i, ] <- colSums(sweep(-Y, 2, -Y[i, ]) * K[, i])
    }
    if (names(exaggeration_factor) == "ls" && iter <= stop_lying_iter) {
      # during LS exaggeration, use gradient descent only with eta = 1
      uY <- -G
    }
    else {
      # compare signs of G with -update (== previous G, if we ignore momentum)
      # abs converts TRUE/FALSE to 1/0
      dbd <- abs(sign(G) != sign(uY))
      gains <- (gains + 0.2) * dbd + (gains * 0.8) * (1 - dbd)
      gains[gains < min_gain] <- min_gain
      uY <- mu * uY - eta * gains * G
    }

    Y <- Y + uY
    # Recenter Y
    Y <- sweep(Y, 2, colMeans(Y))

    if (iter == mom_switch_iter) {
      mu <- final_momentum
      if (verbose) {
        message("Switching to final momentum ", formatC(final_momentum),
                " at iter ", iter)
      }
    }

    if (iter == stop_lying_iter && init != "matrix") {
      if (verbose) {
        message("Switching off exaggeration at iter ", iter)
      }
      P <- P / exaggeration_factor
    }

    if (iter %% epoch == 0 || iter == max_iter) {
      cost <- do_epoch(Y, P, Q, iter, eps, epoch_callback, verbose)

      if (ret_extra) {
        names(cost) <- iter
        itercosts <- c(itercosts, cost)
      }

      if (cost < min_cost) {
        break
      }
    }
  }

  ret_value(Y, ret_extra, X, scale, init, iter, start_time,
            P, Q, eps, perplexity, itercosts,
            stop_lying_iter, mom_switch_iter, momentum, final_momentum, eta,
            exaggeration_factor)
}

# Helper function for epoch callback, allowing user to supply callbacks with
# multiple arities.
do_callback <- function(cb, Y, iter, cost = NULL) {
  nfs <- length(formals(cb))
  if (nfs == 1) {
    cb(Y)
  }
  else if (nfs == 2) {
    cb(Y, iter)
  }
  else if (nfs == 3) {
    cb(Y, iter, cost)
  }
}

# Carry out epoch-related jobs, e.g. cost calculation, logging, callback
do_epoch <- function(Y, P, Q, iter, eps = .Machine$double.eps,
                     epoch_callback = NULL, verbose = FALSE) {
  cost <- sum(P * log((P + eps) / (Q + eps)))

  if (verbose) {
    message(date(), " Iteration #", iter, " error is: ",
            formatC(cost))
  }

  if (!is.null(epoch_callback)) {
    do_callback(epoch_callback, Y, iter, cost)
  }

  cost
}


# Prepare the return value.
# If ret_extra is TRUE, return a list with lots of extra info.
# Otherwise, Y is returned directly.
# If ret_extra is TRUE and iter > 0, then all the NULL-default parameters are
# expected to be present. If iter == 0 then the return list will contain only
# scaling and initialization information.
ret_value <- function(Y, ret_extra, X, scale, init, iter, start_time = NULL,
                      P = NULL, Q = NULL,
                      eps = NULL, perplexity = NULL, itercosts = NULL,
                      stop_lying_iter = NULL, mom_switch_iter = NULL,
                      momentum = NULL, final_momentum = NULL, eta = NULL,
                      exaggeration_factor = NULL) {
  if (ret_extra) {
    end_time <- Sys.time()

    res <- list(
      Y = Y,
      N = nrow(X),
      origD = ncol(X),
      scale = scale,
      init = init,
      iter = iter,
      time_secs = as.numeric(end_time - start_time, units = "secs")
    )

    if (iter > 0) {
      # this was already calculated in the final epoch, but unlikely to be
      # worth the extra coupling and complication of getting it over here
      costs <- colSums(P * log((P + eps) / (Q + eps)))

      if (names(exaggeration_factor) != "ls") {
        names(exaggeration_factor) <- NULL
      }

      res <- c(res, list(
        perplexity = perplexity,
        costs = costs,
        itercosts = itercosts,
        stop_lying_iter = stop_lying_iter,
        mom_switch_iter = mom_switch_iter,
        momentum = momentum,
        final_momentum = final_momentum,
        eta = eta,
        exaggeration_factor = exaggeration_factor
      ))
    }
    res
  }
  else {
    Y
  }
}