library(keras)
library(tensorflow)

# Find the neural network  on unbalanced data ####
# split the data set
set.seed(240)
train_ids = bagofwords_original  %>% group_by(pres_num) %>%
  slice_sample(prop = 0.7) %>%
  ungroup() %>% select(index)

training = bagofwords_original  %>% right_join(train_ids,by ='index') %>% select(-index)
testing  = bagofwords_original  %>% anti_join(train_ids,by ='index') %>% select(-index)

train = list()
test  = list()

train$x = as.matrix(training[,-1])
train$y = as.integer(training$pres_num)

test$x = as.matrix(testing[,-1])
test$y = as.integer(testing$pres_num)

train_y = to_categorical(train$y-1,num_classes = 6)
test_y  = to_categorical(test$y-1,num_classes = 6)

m1 <- keras_model_sequential() %>% 
  layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
  layer_dense(units = 900,activation = 'relu') %>%
  layer_dense(units = 6, activation = 'softmax')

# add drop out for each layer and uses rmsprop
m2 <- keras_model_sequential() %>% 
  layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 900,activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 6, activation = 'softmax')

# extra layer - 1 extra hidden layer and sigmoid
m3 <- keras_model_sequential() %>% 
  layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
  layer_dense(units = 900,activation = 'relu') %>%
  layer_dense(units = 1020,activation = 'sigmoid') %>%
  layer_dense(units = 6, activation = 'softmax')

# same as model 3, but the data will be scaled
m4 <- keras_model_sequential() %>% 
  layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
  layer_dense(units = 900,activation = 'relu') %>%
  layer_dense(units = 1020,activation = 'relu') %>%
  layer_dense(units = 6, activation = 'softmax')

# compile the model
m1 %>% compile(loss = 'categorical_crossentropy',
               optimizer = 'adam', metrics = 'accuracy')
m2 %>% compile(loss = 'categorical_crossentropy',
               optimizer = 'rmsprop', metrics = 'accuracy')
m3 %>% compile(loss = 'categorical_crossentropy',
               optimizer = 'adam', metrics = 'accuracy')
m4 %>% compile(loss = 'categorical_crossentropy',
               optimizer = 'adam', metrics = 'accuracy')

# Train the model
hist1 = m1 %>% fit(train$x,train_y,epochs = 40,batch_size = 5,verbose = 0)
hist2 = m2 %>% fit(train$x,train_y,epochs = 40,batch_size = 5,verbose = 0)
hist3 = m3 %>% fit(train$x,train_y,epochs = 40,batch_size = 5,verbose = 0)
hist4 = m4 %>% fit(scale(train$x),train_y,epochs = 40,batch_size = 5,verbose = 0)

plot(hist4)
r1 = m1 %>% evaluate(test$x,test_y,batch_size = 5,verbose = 2)
r2 = m2 %>% evaluate(test$x,test_y,batch_size = 5,verbose = 2)
r3 = m3 %>% evaluate(test$x,test_y,batch_size = 5,verbose = 2)
r4 = m4 %>% evaluate(scale(test$x),test_y,batch_size = 5,verbose = 2)


results_unbalanced = cbind(r1[2],r2[2],r3[2],r4[2])
colnames(results_unbalanced) = c('m1','m2','m3','m4')
# m1        0.5404 
# m2        0.5281    0.5292
# m3        0.5311   (batch size increased)
# 0.538175 0.5348231 0.5273743 0.5046555 


# Find the neural network  on balanced data ####

set.seed(240)
train_ids = bagofwords_balanced %>% group_by(pres_num) %>%
  slice_sample(prop = 0.7) %>%
  ungroup() %>% select(index)

training = bagofwords_balanced  %>% right_join(train_ids,by ='index') %>% select(-index)
testing  = bagofwords_balanced  %>% anti_join(train_ids,by ='index') %>% select(-index)

train = list()
test  = list()

train$x = as.matrix(training[,-1])
train$y = as.integer(training$pres_num)

test$x = as.matrix(testing[,-1])
test$y = as.integer(testing$pres_num)

train_y = to_categorical(train$y-1,num_classes = 5)
test_y  = to_categorical(test$y-1,num_classes = 5)

mm1 <- keras_model_sequential() %>% 
      layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
      layer_dense(units = 900,activation = 'relu') %>%
      layer_dense(units = 5, activation = 'softmax')

# add drop out for each layer and uses rmsprop
mm2 <- keras_model_sequential() %>% 
        layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(units = 900,activation = 'relu') %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(units = 5, activation = 'softmax')

# extra layer - 1 extra hidden layer and sigmoid
mm3 <- keras_model_sequential() %>% 
      layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
      layer_dense(units = 900,activation = 'relu') %>%
      layer_dense(units = 1020,activation = 'sigmoid') %>%
      layer_dense(units = 5, activation = 'softmax')

# same as model 3, but the data will be scaled
mm4 <- keras_model_sequential() %>% 
      layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
      layer_dense(units = 900,activation = 'relu') %>%
      layer_dense(units = 1020,activation = 'relu') %>%
      layer_dense(units = 5, activation = 'softmax')

# compile the model
mm1 %>% compile(loss = 'categorical_crossentropy',
               optimizer = 'adam', metrics = 'accuracy')
mm2 %>% compile(loss = 'categorical_crossentropy',
               optimizer = 'rmsprop', metrics = 'accuracy')
mm3 %>% compile(loss = 'categorical_crossentropy',
               optimizer = 'adam', metrics = 'accuracy')
mm4 %>% compile(loss = 'categorical_crossentropy',
               optimizer = 'adam', metrics = 'accuracy')

# Train the model
h1 = mm1 %>% fit(train$x,train_y,epochs = 40,batch_size = 5,verbose = 0)
h2 = mm2 %>% fit(train$x,train_y,epochs = 40,batch_size = 5,verbose = 0)
h3 = mm3 %>% fit(train$x,train_y,epochs = 40,batch_size = 5,verbose = 0)
h4 = mm4 %>% fit(scale(train$x),train_y,epochs = 40,batch_size = 5,verbose = 0)


rr1 = mm1 %>% evaluate(test$x,test_y,batch_size = 10,verbose = 2)
rr2 = mm2 %>% evaluate(test$x,test_y,batch_size = 10,verbose = 2)
rr3 = mm3 %>% evaluate(test$x,test_y,batch_size = 10,verbose = 2)
rr4 = mm4 %>% evaluate(scale(test$x),test_y,batch_size = 10,verbose = 2)


results_bal = cbind(rr1[2],rr2[2],rr3[2],rr4[2])
colnames(results_bal) = c('mm1','mm2','mm3','mm4')


# Find the neural network  on tfidf_original ####
set.seed(240)
train_ids = bagofwords_original %>% group_by(pres_num) %>%
  slice_sample(prop = 0.7) %>%
  ungroup() %>% select(index)

training = tfidf_original %>% right_join(train_ids,by ='index') %>% select(-index)
testing  = tfidf_original%>% anti_join(train_ids,by ='index') %>% select(-index)

train = list()
test  = list()

train$x = as.matrix(training)
train$y = as.integer(training$pres_num)

test$x = as.matrix(testing[,-1])
test$y = as.integer(testing$pres_num)

train_y = to_categorical(train$y-1,num_classes = 5)
test_y  = to_categorical(test$y-1,num_classes = 5)

o1 <- keras_model_sequential() %>% 
  layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
  layer_dense(units = 900,activation = 'relu') %>%
  layer_dense(units = 5, activation = 'softmax')

# add drop out for each layer and uses rmsprop
o2 <- keras_model_sequential() %>% 
  layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 900,activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 5, activation = 'softmax')

# extra layer - 1 extra hidden layer and sigmoid
o3 <- keras_model_sequential() %>% 
  layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
  layer_dense(units = 900,activation = 'relu') %>%
  layer_dense(units = 1020,activation = 'sigmoid') %>%
  layer_dense(units = 5, activation = 'softmax')

# same as model 3, but the data will be scaled
o4 <- keras_model_sequential() %>% 
  layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
  layer_dense(units = 900,activation = 'relu') %>%
  layer_dense(units = 1020,activation = 'relu') %>%
  layer_dense(units = 5, activation = 'softmax')

# compile the model
o1 %>% compile(loss = 'categorical_crossentropy',
               optimizer = 'adam', metrics = 'accuracy')
o2 %>% compile(loss = 'categorical_crossentropy',
               optimizer = 'rmsprop', metrics = 'accuracy')
o3 %>% compile(loss = 'categorical_crossentropy',
               optimizer = 'adam', metrics = 'accuracy')
o4 %>% compile(loss = 'categorical_crossentropy',
               optimizer = 'adam', metrics = 'accuracy')

# Train the model
ho1 = o1 %>% fit(train$x,train_y,epochs = 40,batch_size = 5,verbose = 0)
ho2 = o2 %>% fit(train$x,train_y,epochs = 40,batch_size = 5,verbose = 0)
ho3 = o3 %>% fit(train$x,train_y,epochs = 40,batch_size = 5,verbose = 0)
ho4 = o4 %>% fit(scale(train$x),train_y,epochs = 40,batch_size = 5,verbose = 0)


rro1 = o1 %>% evaluate(test$x,test_y,batch_size = 10,verbose = 2)
rro2 = o2 %>% evaluate(test$x,test_y,batch_size = 10,verbose = 2)
rro3 = o3 %>% evaluate(test$x,test_y,batch_size = 10,verbose = 2)
rro4 = o4 %>% evaluate(scale(test$x),test_y,batch_size = 10,verbose = 2)


results_tf_original = cbind(rro1[2],rro[2],rro3[2],rro4[2])
colnames(results_tf_balanced) = c('o1','o2','o3','o4')



# Find the neural network  on tfidf_balanced ####
set.seed(240)
train_ids = bagofwords_balanced %>% group_by(pres_num) %>%
  slice_sample(prop = 0.7) %>%
  ungroup() %>% select(index)

training = tfidf_balanced  %>% right_join(train_ids,by ='index') %>% select(-index)
testing  = tfidf_balanced  %>% anti_join(train_ids,by ='index') %>% select(-index)

train = list()
test  = list()

train$x = as.matrix(training[,-1])
train$y = as.integer(training$pres_num)

test$x = as.matrix(testing[,-1])
test$y = as.integer(testing$pres_num)

train_y = to_categorical(train$y-1,num_classes = 5)
test_y  = to_categorical(test$y-1,num_classes = 5)

f1 <- keras_model_sequential() %>% 
  layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
  layer_dense(units = 900,activation = 'relu') %>%
  layer_dense(units = 5, activation = 'softmax')

# add drop out for each layer and uses rmsprop
f2 <- keras_model_sequential() %>% 
  layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 900,activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 5, activation = 'softmax')

# extra layer - 1 extra hidden layer and sigmoid
f3 <- keras_model_sequential() %>% 
  layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
  layer_dense(units = 900,activation = 'relu') %>%
  layer_dense(units = 1020,activation = 'sigmoid') %>%
  layer_dense(units = 5, activation = 'softmax')

# same as model 3, but the data will be scaled
f4 <- keras_model_sequential() %>% 
  layer_dense(units = 1024,input_shape = c(ncol(train$x)),activation = 'relu') %>%
  layer_dense(units = 900,activation = 'relu') %>%
  layer_dense(units = 1020,activation = 'relu') %>%
  layer_dense(units = 5, activation = 'softmax')

# compile the model
f1 %>% compile(loss = 'categorical_crossentropy',
                optimizer = 'adam', metrics = 'accuracy')
f2 %>% compile(loss = 'categorical_crossentropy',
                optimizer = 'rmsprop', metrics = 'accuracy')
f3 %>% compile(loss = 'categorical_crossentropy',
                optimizer = 'adam', metrics = 'accuracy')
f4 %>% compile(loss = 'categorical_crossentropy',
                optimizer = 'adam', metrics = 'accuracy')

# Train the model
hf1 = f1 %>% fit(train$x,train_y,epochs = 40,batch_size = 5,verbose = 0)
hf2 = f2 %>% fit(train$x,train_y,epochs = 40,batch_size = 5,verbose = 0)
hf3 = f3 %>% fit(train$x,train_y,epochs = 40,batch_size = 5,verbose = 0)
hf4 = f4 %>% fit(scale(train$x),train_y,epochs = 40,batch_size = 5,verbose = 0)


rrf1 = f1 %>% evaluate(test$x,test_y,batch_size = 10,verbose = 2)
rrf2 = f2 %>% evaluate(test$x,test_y,batch_size = 10,verbose = 2)
rrf3 = f3 %>% evaluate(test$x,test_y,batch_size = 10,verbose = 2)
rrf4 = f4 %>% evaluate(scale(test$x),test_y,batch_size = 10,verbose = 2)


results_tf_balanced = cbind(rrf1[2],rr2[2],rrf3[2],rrf4[2])
colnames(results_tf_balanced) = c('f1','f2','f3','f4')

save.image('NN.RData')