##A topic model for musical instrument reviews

In this document, we fit an LDA topic model to the Amazon Review Data of Musical Instruments introduced by Julian McAuley in 2016. To fit the model, we use the R package lda and we visualize the output using LDAvis.

###The data

The data set includes 10,261 reviews and each review includes the ID of the reviewer, ID of the product, Name of the reviewer, Helpfulness rating of the review, e.g. 2/, Text of the review, Rating of the product, Summary of the review, Time of the review (unix time), and Time of the review (raw)

We analyze the Text of the Review in this particular model.

###Pre-processing:

Before fitting a topic model, we need to tokenize the text. This dataset is already fairly clean, so we only remove punctuation and some common and few irrelevant stop words. In particular, we use the english stop words from the SMART information retrieval system, available in the R package tm.

```s
#Cleaning corpus
stop_words <- stopwords("SMART")
stop_words <- c(stop_words,"just", "can", "also", "realli", "thing", "even")
stop_words <- tolower(stop_words)

review <- gsub("'", "", review) # removing apostrophes
review <- gsub("[[:punct:]]", " ", review)  # replacing punctuation with space
review <- gsub("[[:cntrl:]]", " ", review)  # replacing control characters with space
review <- gsub("^[[:space:]]+", "", review) # removing whitespace at beginning of documents
review <- gsub("[[:space:]]+$", "", review) # removing whitespace at end of documents
review <- gsub("[^a-zA-Z -]", " ", review) # allows only letters
review <- tolower(review)  # forcing to lowercase

## getting rid of blank docs
review <- review[review != ""]

# tokenizing on space and output as a list:
doc.list <- strsplit(review, "[[:space:]]+")

# computing the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)


# removing terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
term.table <- term.table[names(term.table) != ""]
vocab <- names(term.table)

# putting the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)
```


###Using the R package 'lda' for model fitting

The object *documents* is a length-10,254 list where each element represents one document, according to the specifications of the lda package. After creating this list, we compute a few statistics about the corpus:

```s
# Computing some statistics related to the data set:
D <- length(documents)  # number of documents (10,254)
W <- length(vocab)  # number of terms in the vocab (5,908)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [16, 31, 36, 17, 13, 20 ...]
N <- sum(doc.length)  # total number of tokens in the data (344,097)
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus [5549, 4121, 3811, 3608, 3423, ...]
```


Next, we set up a topic model with 15 topics, relatively diffuse priors for the topic-term distributions (ηη = 0.02) and document-topic distributions (αα = 0.02), and we set the collapsed Gibbs sampler to run for 3,000 iterations (slightly conservative to ensure convergence). A visual inspection of *fit$log.likelihood* shows that the MCMC algorithm has converged after 3,000 iterations. This block of code takes about 8 minutes to run on a laptop using a single core 2.2Ghz processor (and 8GB RAM).

```s
# MCMC and model tuning parameters:
K <- 20
G <- 3000
alpha <- 0.02
eta <- 0.02

# Fitting the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2 - t1  # about 8 minutes on laptop
```


###Visualizing the fitted model with LDAvis

To visualize the result using LDAvis, we'll need estimates of the document-topic distributions, which we denote by the D×K matrix θ, and the set of topic-term distributions, which we denote by the K×W matrix ϕ. We estimate the “smoothed” versions of these distributions (“smoothed” means that we've incorporated the effects of the priors into the estimates) by cross-tabulating the latent topic assignments from the last iteration of the collapsed Gibbs sampler with the documents and the terms, respectively, and then adding pseudocounts according to the priors. 

```s
theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))
```
We've already computed the number of tokens per document and the frequency of the terms across the entire corpus. We save these, along with ϕ, θ, and vocab, in a list as the data object *reviews_for_LDA*, which is included in the LDAvis package.

```s
reviews_for_LDA <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)
```                    
We now call the *createJSON()* function in **LDAvis**. This function will return a character string representing a JSON object used to populate the visualization. The *createJSON()* function computes topic frequencies, inter-topic distances, and projects topics onto a two-dimensional plane to represent their similarity to each other. It also loops through a grid of values of a tuning parameter, **0≤λ≤10≤λ≤1**, that controls how the terms are ranked for each topic, where terms are listed in decreasing of *relevance*, where the relevance of term *w* to topic *t* is defined as **λ×p(w∣t)+(1−λ)×p(w∣t)/p(w)λ×p(w∣t)+(1−λ)×p(w∣t)/p(w)**. Values of **λ** near 1 give high relevance rankings to *frequent* terms within a given topic, whereas values of **λ** near zero give high relevance rankings to *exclusive* terms within a topic. The set of all terms which are ranked among the top-*R* most relevant terms for each topic are pre-computed by the *createJSON()* function and sent to the browser to be interactively visualized using D3 as part of the JSON object.

```s
library(LDAvis)

# create the JSON object to feed the visualization:
json <- createJSON(phi = reviews_for_LDA$phi, 
                   theta = reviews_for_LDA$theta, 
                   doc.length = reviews_for_LDA$doc.length, 
                   vocab = reviews_for_LDA$vocab, 
                   term.frequency = reviews_for_LDA$term.frequency)
```                   
                   
The *serVis()* function can take *json* and serve the result in a variety of ways. Here we write *json* to a file within the 'Amzon_Reviews' directory (along with other HTML and JavaScript required to render the page). 

```s
serVis(json, out.dir = 'Amazon_Reviews', open.browser = TRUE)
```

The result can be seen [here] (https://htmlpreview.github.io/?https://github.com/ushnik/Amazon_Reviews/blob/master/index.html)

Hovering over different topic numbers gives us the terms and their frequency of usage. This changes with the relevce setting. For example, when we look at the 30 most relevant terms for Topic 4 using a relevance setting of λ=0.5, the term “strings” is the 1st bar from the top (i.e. the most relevant term for this topic). The widths of the red and blue bars indicate that there is at least one other topic in which the term “action” appears frequently. By hovering over “strings”, we see from the following state of **LDAvis** that term “strings” also appears frequently in Topic 8 (as the 13th most relevant term):

https://htmlpreview.github.io/#topic=8&lambda=0.5&term=

Comparing these two topics, we can see that Topic 4 discusses strings in the context of acoustic guitar strings and companies like Daddario and Elixir that sell them, whereas in Topic 8, the term “strings”“ is also used frequently, but the topic is specifically about stringed instruments like guitars, ukulele, violin, etc. These two topics both make use of the word "strings” but in slightly different contexts.



