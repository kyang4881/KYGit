### Clustering Iris Dataset ###

df <- iris
df <- na.omit(df)

#Convert categorical to numeric
df <- cbind(df[,1:4], Species = transform(as.numeric(df$Species)))

df <- scale(df)
distance <- get_dist(df)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))
k2 <- kmeans(df, centers = 2, nstart = 25)
str(k2)
k2
fviz_cluster(k2, data = df)

k3 <- kmeans(df, centers = 3, nstart = 25)
k4 <- kmeans(df, centers = 4, nstart = 25)
k5 <- kmeans(df, centers = 5, nstart = 25)

#Plots to compare
p1 <- fviz_cluster(k2, geom = "point", data = df) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = df) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = df) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = df) + ggtitle("k = 5")

grid.arrange(p1, p2, p3, p4, nrow = 2)

set.seed(123)

#Compute total within-cluster sum of squares 
wss <- function(k) {
  kmeans(df, k, nstart = 10 )$tot.withinss
}

#Compute and plot wss for k = 1 to k = 15
k.values <- 1:15

#Extract wss for 2-15 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")


fviz_nbclust(df, kmeans, method = "wss")

avg_sil <- function(k) {
  km.res <- kmeans(df, centers = k, nstart = 25)
  ss <- silhouette(km.res$cluster, dist(df))
  mean(ss[, 3])
}

#Compute and plot wss for k = 2 to k = 15
k.values <- 2:15

#Extract avg silhouette for 2-15 clusters
avg_sil_values <- map_dbl(k.values, avg_sil)

plot(k.values, avg_sil_values,
     type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of clusters K",
     ylab = "Average Silhouettes")

fviz_nbclust(df, kmeans, method = "silhouette")

gap_stat <- clusGap(df, FUN = kmeans, nstart = 25,
                    K.max = 10, B = 50)
#Print results
print(gap_stat, method = "firstmax")

fviz_gap_stat(gap_stat)

final <- kmeans(df, 3, nstart = 25)
print(final)

fviz_cluster(final, data = df)
