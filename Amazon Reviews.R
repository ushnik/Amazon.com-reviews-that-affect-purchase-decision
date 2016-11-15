library(quanteda)
library(stm)
library(tm)
library(NLP)
library(openNLP)
library(ggplot2)
library(ggdendro)
library(cluster)
library(fpc)  


##Loading Json file##

library(ndjson)
data<- ndjson::stream_in("C:/Users/Ushnik/Desktop/Musical_Instruments_5.json")
data
View(data)

#####################################################################################

#Generating DFM

require(quanteda)

corpus<-toLower(data$reviewText, keepAcronyms = F )
cleancorpus <- tokenize(corpus, 
                        removeNumbers=TRUE, 
                        removePunct = TRUE, 
                        removeSeparators=TRUE, 
                        removeTwitter=TRUE, 
                        verbose=TRUE)

dfm<- dfm(cleancorpus,
          toLower = TRUE, 
          ignoredFeatures =stopwords("english"), 
          verbose=TRUE, 
          stem=TRUE)

topfeatures(dfm, 50)     # displaying 50 features

###########################################################################################

## Hierc. Clustering## 

dfm.tm<-convert(dfm, to="tm")
dfm.tm
dtmss <- removeSparseTerms(dfm.tm, 0.85)
dtmss
d.dfm <- dist(t(dtmss), method="euclidian")
fit <- hclust(d=d.dfm, method="average")
hcd <- as.dendrogram(fit)

require(cluster)
k<-5
plot(hcd, ylab = "Distance", horiz = FALSE, 
     main = "Five Cluster Dendrogram", 
     edgePar = list(col = 2:3, lwd = 2:2))
rect.hclust(fit, k=k, border=1:5) # drawing dendogram with red borders around the 5 clusters

ggdendrogram(fit, rotate = TRUE, size = 4, theme_dendro = FALSE,  color = "blue") +
  xlab("Features") + 
  ggtitle("Cluster Dendrogram")

require(fpc)   
d <- dist(t(dtmss), method="euclidian")   
kfit <- kmeans(d, 5)   
clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=2, lines=0)

#################################################################################################

##Topic Modelling##

library(dplyr)
require(magrittr)
library(tm)
library(ggplot2)
library(stringr)
library(NLP)
library(openNLP)

review<-data$reviewText 

#Cleaning corpus
stop_words <- stopwords("SMART")
stop_words <- c(stop_words,"just", "can", "also", "realli", "thing", "even")
stop_words <- tolower(stop_words)

review <- gsub("'", "", review) # remove apostrophes
review <- gsub("[[:punct:]]", " ", review)  # replace punctuation with space
review <- gsub("[[:cntrl:]]", " ", review)  # replace control characters with space
review <- gsub("^[[:space:]]+", "", review) # remove whitespace at beginning of documents
review <- gsub("[[:space:]]+$", "", review) # remove whitespace at end of documents
review <- gsub("[^a-zA-Z -]", " ", review) # allows only letters
review <- tolower(review)  # force to lowercase

## get rid of blank docs
review <- review[review != ""]

# tokenize on space and output as a list:
doc.list <- strsplit(review, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)


# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
term.table <- term.table[names(term.table) != ""]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

#############
# Compute some statistics related to the data set:
D <- length(documents)  
W <- length(vocab)  
doc.length <- sapply(documents, function(x) sum(x[2, ]))  
N <- sum(doc.length)  
term.frequency <- as.integer(term.table) 

# MCMC and model tuning parameters:
K <- 15
G <- 3000
alpha <- 0.02
eta <- 0.02

# Fit the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
## display runtime
t2 - t1  

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

reviews_for_LDA <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)

library(LDAvis)
library(servr)

# create the JSON object to feed the visualization:
json <- createJSON(phi = reviews_for_LDA$phi, 
                   theta = reviews_for_LDA$theta, 
                   doc.length = reviews_for_LDA$doc.length, 
                   vocab = reviews_for_LDA$vocab, 
                   term.frequency = reviews_for_LDA$term.frequency)

serVis(json, out.dir = 'Amazon_Reviews', open.browser = TRUE)
