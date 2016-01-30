library("readr")
library("stringr")
library("yaml")
library("dplyr")
# Load master.yaml
### The yaml library does not implement a way to read a yaml file that contains
### multiple documents. One solution is to read in master.yaml as one continuous
### string, split the string at the document separater, and then parse each
### document into a list in turn.
###
### After parsing the documents, unnecessary fields can be dropped and, if all
### of the list elements are singletons, the list can be converted into a
### dataframe for easy of use.
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
ystr <- readr::read_file("./manchester/master.yaml")
ylst <- stringr::str_split(ystr,"---")[[1]]
db <- lapply(ylst, yaml::yaml.load)
db <- lapply(db, dropfields, c("ATLASDist","COPY","PythonDist","URLS","executable","libfiles","verbose","wrapper","writemode"))
db <- do.call(rbind.data.frame, db)
db$job <- 0:(nrow(db)-1)
db$condition <- as.factor(ifelse("Images_judge_kind/responses.csv"==db$responses,"kind","size"))
str(db)

r <- read.csv("./manchester/shared/Images_judge_size/responses.csv")
labels <- unique(sort(r$Center))
tmp <- sapply(stringr::str_split(labels,"_"),"[",2)
labels <- sapply(stringr::str_split(tmp,"\\."),"[",1)

# Load data
fmt <- sprintf("manchester/%%0%dd", nchar(as.character(nrow(db))))
db_select <- filter(db, proportion==1.0, condition=="size", ndim==2, epsilon==1e-7)
models <- lapply(db_select$job, loadfromjob, "model.csv", fmt, header=F)
db_select
nrow(models[[1]])

m <- models[[1]]
rownames(m) <- labels

png("NEXT_Manchester37_size.png", width=700, height=700)
plot(m)
text(m, labels = labels)
mtext("NEXT Manchester Line-drawing Size Embedding",line = 0.5,cex=1.5)
dev.off()

write.csv(m, file="NEXT_Manchester37_size_2D.csv")

