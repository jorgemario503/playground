##Load data
library(readxl)
df <- data.frame(read_excel("Pharmaceuticals.xls"))
summary(df)
head(df,12)
# normalization of the dataset
df.n <- sapply(df[,c(3:11)], scale)
# add row names
row.names(df.n) <- df[,1]

##Exploratory correlation and PCA
# compute the correlation matrix and draw the heatmap
cor(df.n)
heatmap(cor(df.n), Rowv = NA, Colv = NA)
# view principal components
pcs <- prcomp(df.n)
summary(pcs)
# view top principal components
pcs$rotation[, 1:5] 
#scatterplot of PC1 vs. PC2 scores
plot(x = pcs$x[, 1], y = pcs$x[, 2], xlab = "PC1", ylab = "PC2")
#color-coded by type
text(x = pcs$x[, 1], y = pcs$x[, 2], labels = df[,3], pos=1) 

##Cluster analysis for quantitative variables
# denogram euclidean and manhattan distance
euclidean <- dist(df.n[,-c(1)], method = "euclidean")
manhattan <- dist(df.n[,-c(1)], method = "manhattan")
# denogram single linkage
hc1 <- hclust(euclidean, method = "single")
plot(hc1, hang = -1, ann = FALSE)
hc2 <- hclust(manhattan, method = "single")
plot(hc2, hang = -1, ann = FALSE)
# denogram complete linkage
hc3 <- hclust(euclidean, method = "complete")
plot(hc3, hang = -1, ann = FALSE)
hc4 <- hclust(manhattan, method = "complete")
plot(hc4, hang = -1, ann = FALSE)

##Aggregation comparison by mean
cluster4 <- cutree(hc3, k = 4)
table(cluster4)
c4 = aggregate(df[,c(3:11)],by=list(cluster4),FUN=mean, na.rm=TRUE)
c4
# plot heatmap - rev() reverses the color mapping to large = dark
heatmap(as.matrix(c4[,-1]), Colv=NA,col=rev(paste("gray",1:99,sep="")))
# scatterplot by cluster
df.x <- data.frame(cluster4)[,1]
df.y <- data.frame(df)[,c(2,12:14)]
df.y$other <- paste(df.y$Location,df.y$Exchange)
plot(x = df.x, y = as.factor(df.y[,5]), xlab = "Cluster", ylab = "Dimension")
text(x = df.x, y = as.factor(df.y[,5]), labels = df.y[,5], col=hcl.colors(4,"Hawaii")[df.x], pos=1) 
