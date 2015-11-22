# rtsne

A fork of Justin Donaldson's R package for t-SNE (t-Distributed Stochastic 
Neighbor Embedding).

I just wanted to teach myself how TSNE worked, while also learning non-trivial 
and idiomatic R programming. No particular reason to use this fork, unless you
are interested in debugging alternative implementations, in which case, this
has the advantage of adding an initialization from PCA option, so you can
avoid random initialization.

### Changes

I made the following minor changes to make the output more reproducible and 
easier to compare and debug with other implementations:

* Added an option to initialize from the first two PCA scores.
* You can specify the early exaggeration, momentum and learning rate parameters.
* Minor formatting changes to somewhat appease R studio diagnostics and `lintr`.
* Documented the functions with Roxygen.

### Installing:
```R
install.packages("devtools")
devtools::install_github("jlmelville/rtsne/tsne")
```

### Using:
```R
iris_plot <- function() {
  colors = rainbow(length(unique(iris$Species)))
  names(colors) = unique(iris$Species)
  function(x,y) {
    plot(x, t = 'n'); text(x, labels = iris$Species, col = colors[iris$Species])
  }
}

# whitens data by default, may not be what you want for non-imaging data
tsne_iris_whitened <- tsne::tsne(iris[,1:4], perplexity = 25, 
                                 epoch_callback = iris_plot())

# use input data as-is
tsne_iris <- tsne::tsne(iris[,1:4], perplexity = 25, epoch_callback = iris_plot(), 
                        whiten = FALSE)

# use PCA initialization so embedding is repeatable
tsne_iris_pca <- tsne::tsne(iris[,1:4], perplexity = 25, epoch_callback = iris_plot(),
                       whiten = FALSE, init_from_PCA = TRUE)
```

### License
[GPLv2 or later](https://www.gnu.org/licenses/gpl-2.0.txt).
