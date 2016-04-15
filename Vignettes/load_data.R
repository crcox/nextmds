library("readr")
library("stringr")
library("yaml")
library("dplyr")
library("ggplot2")
library("igraph")
source("~/src/nextmds/Vignettes/model_error.R")
# Load master.yaml
### The yaml library does not implement a way to read a yaml file that contains
### multiple documents. One solution is to read in master.yaml as one continuous
### string, split the string at the document separater, and then parse each
### document into a list in turn.
###
### After parsing the documents, unnecessary fields can be dropped and, if all
### of the list elements are singletons, the list can be converted into a
### dataframe for easy of use.
MASTER <- "./Foods/master.yaml"
FIELDS_TO_DROP <- c("ATLASDist","COPY","PythonDist","URLS","executable","libfiles","verbose","wrapper","writemode")
dropfields <- function(x, fieldnames) {
  for ( f in fieldnames ) {
    x[[f]] <- NULL
  }
  return(x)
}
loadfromjob <- function(x, filename, fmt, ...) {
  fpath <- file.path(sprintf(fmt,x),filename)
  d <- read.csv(fpath, ...)
  return(d)
}
ystr <- readr::read_file(MASTER)
ylst <- stringr::str_split(ystr,"---")[[1]]
db <- lapply(ylst, yaml::yaml.load)
db <- lapply(db, dropfields, FIELDS_TO_DROP)
db <- do.call(rbind.data.frame, db)
db$job <- 0:(nrow(db)-1)
db$condition <- as.factor(ifelse("data/queries_men.csv"==db$responses,"male","female"))
str(db)

r <- read.csv("./Foods/shared/data/queries_men.csv")
labels <- unique(sort(r$Center))
#tmp <- sapply(stringr::str_split(labels,"_"),"[",2)
#labels <- sapply(stringr::str_split(tmp,"\\."),"[",1)

# Load data
fmt <- sprintf("./Foods/%%0%dd", nchar(as.character(nrow(db)-1)))
db_select <- filter(db, proportion==1.0)
Model <- lapply(db_select$job, loadfromjob, "model.csv", fmt, header=F)
TestSet <- lapply(db_select$job, loadfromjob, "responses_Test_TEST.csv", fmt, header=F)
TrainSet <- lapply(db_select$job, loadfromjob, "responses_Uncertainty_TRAIN.csv", fmt, header=F)
db_select
nrow(Model[[1]])

tmp1 <- db_select[,c('condition','ndim')]
tmp1$testOn <-factor(1,levels=1:2,labels=c('male','female'))
tmp1$test <- NA
tmp1$train <- NA

tmp2 <- db_select[,c('condition','ndim')]
tmp2$testOn <-factor(2,levels=1:2,labels=c('male','female'))
tmp2$test <- NA
tmp2$train <- NA

Error <- rbind(tmp1,tmp2)
rm('tmp1','tmp2')

for (iError in 1:nrow(Error)) {
  i <- which(db_select$ndim == Error$ndim[iError] & db_select$condition == Error$condition[iError])
  j <- which(db_select$ndim == Error$ndim[iError] & db_select$condition == Error$testOn[iError])
  Error$test[iError] <- model_error(Model[[i]],TestSet[[j]])
  Error$train[iError] <- model_error(Model[[i]],TrainSet[[j]])
}
Error$congruent <- factor(Error$condition == Error$testOn, levels=c(T,F), labels=c("Test on modeled gender","Test on unmodeled gender"))
ggplot(Error,aes(x=ndim, color=condition)) +
  geom_line(aes(y=test), size=2, linetype=1) +
  #geom_line(aes(y=train), size=2, linetype=2) +
  ggtitle("Foods") +
  theme_grey(base_size=18) +
  facet_wrap("congruent")

CombinationIndex <- as.data.frame(t(combn(451,2)))
CombinationIndex[,1] <- labels[CombinationIndex[,1]]
CombinationIndex[,2] <- labels[CombinationIndex[,2]]
names(CombinationIndex) <- c("from","to")
CombinationIndex$weight <- NA
Edge <- list(male=CombinationIndex,female=CombinationIndex)
Edge$male$weight <- abs(cosineDist(as.matrix(Model[[9]])))
Edge$female$weight <- abs(cosineDist(as.matrix(Model[[10]])))

Graph <- list(
  male=graph_from_data_frame(Edge$male, directed = FALSE, vertices=data.frame(name=labels)),
  female=graph_from_data_frame(Edge$female, directed = FALSE, vertices=data.frame(name=labels))
)

GENERIC <- str_detect(labels, "food")|str_detect(labels, "alcohol")|str_detect(labels, "dessert")|str_detect(labels, "\\bmeats?\\b")
UNCOMMON <- labels %in% c("eye of the round","oleo","bear","lutefisk","carbohydrate","cold cuts")
Graph$male <- delete_vertices(Graph$male, which(GENERIC|UNCOMMON))
Graph$female <- delete_vertices(Graph$female, which(GENERIC|UNCOMMON))

Graph$male <- delete_edges(Graph$male, which(abs(edge_attr(Graph$male,"weight"))>2))
tmp <- cluster_louvain(Graph$male)
tmp

rownames(m) <- labels

png("NEXT_Manchester37_size.png", width=700, height=700)
plot(m)
text(m, labels = labels)
mtext("NEXT Manchester Line-drawing Size Embedding",line = 0.5,cex=1.5)
dev.off()

write.csv(m, file="NEXT_Manchester37_size_2D.csv")

