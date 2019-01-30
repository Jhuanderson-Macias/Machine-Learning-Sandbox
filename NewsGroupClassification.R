rm(list=ls()); cat("\014") # Clear Workspace and Console
library(tm) # Load the Text Mining package

library(tm)
library(SnowballC)
library(class)S


Doc1.TestPath <- system.file('texts/20newsgroups',package = "tm")


# get wd for each file location
Training.source.1<-DirSource(file.path(Doc1.TestPath, "20news-bydate-train","sci.space"))                                 
Testing.source.1<-DirSource(file.path(Doc1.TestPath,"20news-bydate-test","sci.space"))
Training.source.2<-DirSource(file.path(Doc1.TestPath,"20news-bydate-train","rec.autos"))
Testing.source.2<-DirSource(file.path(Doc1.TestPath,"20news-bydate-test","rec.autos"))

#read 100 files from each file 
#For sci.space
Train.moto<-URISource(Training.source.1$filelist[1:100])
Test.moto<- URISource(Testing.source.1$filelist[1:100])

#For rec.autos
Train.auto<-URISource(Training.source.2$filelist[1:100])
Test.auto<-URISource(Testing.source.2$filelist[1:100])

#  Obtain the merged Corpus (of 400 documents)
#Merge.corpus<-c(Train.moto,Test.moto,Train.auto,Test.auto)
corpus.train.moto<-VCorpus(Train.moto)
corpus.test.moto<-VCorpus(Test.moto)
corpus.train.auto<-VCorpus(Train.auto)
corpus.test.auto<-VCorpus(Test.auto)
Merge.corpus <- c(corpus.train.moto,corpus.test.moto,corpus.train.auto,corpus.test.auto)

inspect(Merge.corpus) 
class(Merge.corpus)
#examine content
Merge.corpus[[400]]
Merge.corpus[[24]]$content[1] # Displays 
#Doc.Merge[[2]]$meta # Editable Metadata 
#corpus.txt[[1]]$meta$author <-'anything'

# Pre-Processing Corupus

list.preprocessing <- c(removePunctuation,removeNumbers,stripWhitespace, stemDocument, content_transformer(tolower),
                        removeWords,stopwords("english"))

processing_function <- function(list.of.PreProcessing, Corpus ){
  for (i in list.preprocessing){
    Corpus<-tm_map(Corpus, removeNumbers)
  }
  return(Corpus)
}

Merge.corpus<-processing_function(list.preprocessing,Merge.corpus)

#Document Term Matrix -- word lengths of at least 2, word frequency of at least 5)
Matrix.corpus<-DocumentTermMatrix(Merge.corpus,control=list(minWordLength=c(2,Inf),
                                                            bounds=list(minDocFreq=c(5,Inf))))
inspect(Matrix.corpus)

#Split the Document-Term Matrix into train dataset and test dataset
Doc.Train<-as.matrix(Matrix.corpus[c(1:100,201:300),])
Doc.Test<-as.matrix(Matrix.corpus[c(101:200,301:400),])

# tag factor
tag.sci <-rep("Rec",100)
tag.rep <- rep("sci",100)
Tag<-factor(c(tag.rep,tag.sci))
# check level
table(Tag)
#relevel
Tag <- relevel(Tag, "sci")
table(Tag)

#KNN Classification
set.seed(0)
prob.test<-knn(Doc.Train, Doc.Test, Tag, k=2,prob=TRUE)

# Display Classification Results
a <- 1:length(prob.test)
b <- levels(prob.test)[prob.test]
c <- attributes(prob.test)$prob
d<-prob.test==Tag
result <- data.frame(Doc=a, Predict=b,Prob=c,Correct=d)
result
# Overall Prob
sum(c)/length(Tag)


sum(prob.test==Tag)/length(Tag) * 100  # % Correct Classification

#Estimate the effectiveness of your classification:

#Confusion Matrix
TP<-sum(prob.test[1:100]==Tag[1:100])
FP<-sum(prob.test[1:100]!=Tag[1:100])
FN<-sum(prob.test[101:200]!=Tag[101:200])
TN<-sum(prob.test[101:200]==Tag[101:200])
table(prob.test,Tag)

Precision<-(TP/(TP+FP))*100
Recall<-(TP/(TP+FN))*100
F.Score<-2*Precision*Recall/(Precision+Recall)

