# rtsne

A fork of Justin Donaldson's R package for t-SNE (t-Distributed Stochastic 
Neighbor Embedding).

I just wanted to teach myself how t-SNE worked, while also learning non-trivial 
and idiomatic R programming. I have subsequently messed about with various
parameters, exposing different options, and also added:

* Extra initialization option: use the first two PCA scores. Makes embedding deterministic. 
This can be scaled so the standard deviation is 1e-4 (as in the usual random initialization).
* Early exaggeration option: the method suggested by [Linderman and Steinerberger](https://arxiv.org/abs/1706.02582).

## Installing:

```R
install.packages("devtools")
devtools::install_github("jlmelville/rtsne/tsne")
library(tsne)
```

## Using:

```R
uniq_spec <- unique(iris$Species)
colors <- rainbow(length(uniq_spec))
names(colors) <- uniq_spec
iris_plot <- function(x) {
  plot(x, col = colors[iris$Species])
}

# set verbose = TRUE to log progress to the console
tsne_iris <- tsne(iris[, -5], perplexity = 25, epoch_callback = iris_plot,
                  verbose = TRUE)

# use (scaled) PCA initialization so embedding is repeatable
tsne_iris_spca <- tsne(iris[, -5], perplexity = 25, epoch_callback = iris_plot, init = "spca")

# whitening
tsne_iris_whiten <- tsne(iris[, -5], perplexity = 25, epoch_callback = iris_plot)

# Dataset-dependent exaggeration suggested by Linderman and Steinerberger
tsne_iris_ls <- tsne(iris[, -5], perplexity = 25, epoch_callback = iris_plot, exaggerate = "ls")
```

## License

[GPLv2 or later](https://www.gnu.org/licenses/gpl-2.0.txt).
