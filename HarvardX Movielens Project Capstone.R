library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# rating

edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_segment( aes(x=rating, xend=rating, y=0, yend=count)) +
  geom_point( size=5, color="red", fill=alpha("orange", 0.3), alpha=0.7, shape=21, stroke=2) +
  xlab("Rating") +
  ylab("Ratings Frequency")

# user effect
edx %>%
  group_by(userId) %>%
  summarize(b_u=mean(rating)) %>%
  ggplot(aes(b_u))+
  geom_histogram(bins=20,color="grey")

# movie effect
mu<-mean(edx$rating)
movie_avg<-edx %>%
  group_by(movieId) %>%
  summarize(b_i=mean(rating-mu))
qplot(b_i,data=movie_avg,bins=10,color=I("grey"))

movie_top<-edx %>% group_by(title) %>% summarize(n=n(),avg=mean(rating)) %>% arrange(desc(n)) %>%
  top_n(10,n)
knitr::kable(movie_top)

#genre effect
edx %>%
  group_by(genres) %>%
  summarize(n = n(),avg = mean(rating),se=sd(rating)/sqrt(n)) %>% filter(n>300000) %>%
  mutate(genres=reorder(genres,avg)) %>%
  ggplot(aes(x=genres,y=avg,ymin=avg-2*se,ymax=avg+2*se))+geom_point()+geom_errorbar()

genre_top<-edx %>%
  group_by(genres) %>%
  summarize(n = n(),avg=mean(rating)) %>% arrange(desc(n)) %>% top_n(10,n)

knitr::kable(genre_top)

genre_effect <- edx %>%
  group_by(genres) %>%
  summarise(c = mean(rating - mu))

# time effect
library(lubridate)
edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "month")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Timestamp, time unit : month")

time_avg<-edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "month")) %>%
  group_by(date) %>%
  summarize(b_t = mean(rating-mu))

qplot(b_t,data=time_avg,bins=10,color=I("grey"))
  

# Model analysis

#1.movie effect

#calculate the average of all ratings of the edx set
mu <- mean(edx$rating)

#calculate b_i on the training set
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

#predicted ratings
predicted_ratings_bi <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

#2.movie + user effect

#calculate b_u using the training set 
user_avgs <- edx %>%  
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#predicted ratings
predicted_ratings_bu <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

#3.movie + user + time effect

#create a copy of validation set, validate, and create the date feature which is the timestamp converted to a datetime object  and  rounded by month.

validate <- validation
validate <- validate %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "month")) 

#calculate time effects (b_t) using the training set
temp_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "month")) %>%
  group_by(date) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u))

#predicted ratings
predicted_ratings_bt <- validate %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(temp_avgs, by='date') %>%
  mutate(pred = mu + b_i + b_u + b_t) %>%
  .$pred

#4.movie + user + time effect + genre effect

#calculate genre effect (b_g) using the training set
genre_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "month")) %>%
  left_join(temp_avgs,by='date') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_t))

#predicted ratings
predicted_ratings_bg <- validate %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(temp_avgs, by='date') %>%
  left_join(genre_avgs,by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
  .$pred

#Root Mean Square Error Loss Function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Model RMSE results - comparison
rmse_model1 <- round(RMSE(validation$rating,predicted_ratings_bi),6) 
rmse_model2 <- round(RMSE(validation$rating,predicted_ratings_bu),6) 
rmse_model3 <- round(RMSE(validation$rating,predicted_ratings_bt),6) 
rmse_model4 <- round(RMSE(validation$rating,predicted_ratings_bg),6) 
rmse_results<-data.frame(method=c("movie effect","movie + user effect",
                                  "movie + user + time effect",
                                  "movie + user + time + genre effect"),
                                  rmse=c(rmse_model1,rmse_model2,rmse_model3,rmse_model4))

knitr::kable(rmse_results)

#Regolarization model
#Partition for the cross-validation and lambda calculation
set.seed(7, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(7)`
test_valid <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_reg <- edx[-test_valid,]
temp <- edx[test_valid,]
# Make sure userId and movieId in test_reg set are also in train_reg set
test_reg <- temp %>% 
  semi_join(train_reg, by = "movieId") %>%
  semi_join(train_reg, by = "userId")

# Add rows removed from test_reg set back into train_reg set
removed <- anti_join(temp, test_reg)
train_reg <- rbind(train_reg, removed)

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas,function(l){
  
  #Calculate the mean of ratings from the training set for cross validation
  mu_reg <- mean(train_reg$rating)
  
  #Adjust mean by movie effect and penalize low number of ratings
  b_i_reg <- train_reg %>% 
    group_by(movieId) %>%
    summarize(b_i_reg = sum(rating - mu_reg)/(n()+l))
  
  #ajdust mean by user and movie effect and penalize low number of ratings
  b_u_reg <- train_reg %>% 
    left_join(b_i_reg, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_reg = sum(rating - b_i_reg - mu_reg)/(n()+l))
  
  #predict ratings in the test set (test_reg) to derive optimal penalty value 'lambda'
  predicted_ratings <- 
    test_reg %>% 
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    mutate(pred = mu_reg + b_i_reg + b_u_reg) %>%
    .$pred
  
  return(RMSE(test_reg$rating, predicted_ratings))
})

qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]

pred_reg <- sapply(lambda,function(l){
  
  #Derive the mean from the training set
  mu <- mean(edx$rating)
  
  #Calculate movie effect with optimal lambda
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  #Calculate user effect with optimal lambda
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  #Predict ratings on validation set
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred #validation
  
  return(predicted_ratings)
  
})

rmse_model5 <- round(RMSE(validation$rating,pred_reg),6) 
new_model<-c("Regularized Movie + User effect Model",rmse_model5)
rmse_results<-rbind(rmse_results,new_model)
knitr::kable(rmse_results)

# unsupervised methods to explore the structure of the data

top<-edx %>%
  group_by(movieId) %>%
  summarize(n=n(),title=first(title)) %>%
  top_n(50,n) %>%
  pull(movieId)

x<-edx %>%
  filter(movieId %in% top) %>%
  group_by(userId) %>%
  filter(n() >=25) %>%
  ungroup() %>%
  select(title,userId,rating) %>%
  spread(userId,rating)

row_names<-str_remove(x$title,": Episode") %>% str_trunc(20)
x<-x[,-1] %>% as.matrix()
x<-sweep(x,2,colMeans(x,na.rm=TRUE))
x<-sweep(x,1,rowMeans(x,na.rm=TRUE))
rownames(x)<-row_names

d<-dist(x)
h<-hclust(d)
plot(h,cex=0.5,main="",xlab="")
groups<-cutree(h,k=10)
names(groups)[groups==3]
names(groups)[groups==10]
names(groups)[groups==5]
library(matrixStats)
sds<-colSds(x,na.rm=TRUE)
o<-order(sds,decreasing=TRUE)[1:25]
heatmap(x[,o],col=RColorBrewer::brewer.pal(11,"RdBu"))

# Recosystem model
## https://statr.me/2016/07/recommender-system-using-parallel-matrix-factorization/

library(recosystem)
train_data<-with(edx,data_memory(user_index = userId,
                                       item_index = movieId,
                                       rating=rating))
validation_data<-with(validation,data_memory(user_index=userId,
                                             item_index=movieId,
                                             rating=rating))

r=Reco()               
r$train(train_data, opts = list(dim = 20,                        
                               costp_l1 = 0, costp_l2 = 0.01,   
                               costq_l1 = 0, costq_l2 = 0.01,   
                               niter = 10,                      
                               nthread = 4)) 

pred = r$predict(validation_data, out_memory())

rmse_model6 <- round(RMSE(validation$rating,pred),6)
new_model2<-c("Recosystem model",rmse_model6)
rmse_results<-rbind(rmse_results,new_model2)
knitr::kable(rmse_results)

# The following code takes a few minutes to run
# opts_tune = r$tune(train_data,
#                               opts = list(dim      = c(10, 20, 30),       
#                               costp_l2 = c(0.01, 0.1),        
#                               costq_l2 = c(0.01, 0.1),        
#                               costp_l1 = 0,                   
#                               costq_l1 = 0,                   
#                               lrate    = c(0.01, 0.1),         
#                               nthread  = 4,                    
#                               niter    = 10,                   
#                               verbose  = TRUE))                
# r$train(train_data, opts = c(opts_tune$min,                     
#                               niter = 100, nthread = 4))          


