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

# By default, we use all numeric columns found in a data frame, so you don't
# need to filter out factor or strings
tsne_iris <- tsne(iris, perplexity = 25, epoch_callback = iris_plot)

# set verbose = TRUE to log progress to the console
tsne_iris <- tsne(iris, perplexity = 25, epoch_callback = iris_plot, verbose = TRUE)

# use (scaled) PCA initialization so embedding is repeatable
tsne_iris_spca <- tsne(iris, perplexity = 25, epoch_callback = iris_plot, init = "spca")

# whitening
tsne_iris_whiten <- tsne(iris[, -5], perplexity = 25, epoch_callback = iris_plot,
                         whiten = TRUE)

# dataset-dependent exaggeration suggested by Linderman and Steinerberger
tsne_iris_ls <- tsne(iris[, -5], perplexity = 25, epoch_callback = iris_plot, exaggeration_factor = "ls")
```

## License

[GPLv2 or later](https://www.gnu.org/licenses/gpl-2.0.txt).
