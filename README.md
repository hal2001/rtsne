# rtsne

A fork of Justin Donaldson's R package for t-SNE (t-Distributed Stochastic 
Neighbor Embedding).

I just wanted to teach myself how t-SNE worked, while also learning non-trivial 
and idiomatic R programming. I have subsequently messed about with various
parameters, exposing different options, and also added:

* Extra initialization option: use the first two PCA scores. Makes embedding deterministic.
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

tsne_iris <- tsne(iris[, -5], perplexity = 25, epoch_callback = iris_plot)

# use PCA initialization so embedding is repeatable
tsne_iris_pca <- tsne(iris[, -5], perplexity = 25, epoch_callback = iris_plot, init_from_PCA = TRUE)

# whitening
tsne_iris_whiten <- tsne(iris[, -5], perplexity = 25, epoch_callback = iris_plot)

# Dataset-dependent exaggeration suggested by Linderman and Steinerberger
tsne_iris_ls <- tsne(iris[, -5], perplexity = 25, epoch_callback = iris_plot, exaggerate = "ls")
```

## License

[GPLv2 or later](https://www.gnu.org/licenses/gpl-2.0.txt).
