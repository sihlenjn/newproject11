library(rpart)

#=========================================================
# CLASSIFICATION TREE - bag of words
#=========================================================
# split the data set
set.seed(240)
train_ids = bagofwords  %>% group_by(pres_num) %>%
                            slice_sample(prop = 0.7) %>%
                            ungroup() %>% select(index)

training = bagofwords  %>% right_join(train_ids,by ='index') %>% select(-index)
testing  = bagofwords  %>% anti_join(train_ids,by ='index') %>% select(-index)

# Fit the model
fit = rpart(pres_num ~ ., training, method = 'class')

# train accuracy
fittedtrain <- predict(fit,type='class')
predtrain   <-  table(training$pres_num,fittedtrain)
# train_accuracy = round(sum(diag(predtrain))/sum(predtrain),3) 

# test accuracy
fittedtest = predict(fit , newdata = testing, type = 'class')
predtest  = table(testing$pres_num,fittedtest)
(test_accuracy = round(sum(diag(predtest))/sum(predtest),3))

# Experiment with method used to deal with imbalanced data


#=========================================================
# CLASSIFICATION TREE - it-dft  format 
#=========================================================
training_tf = tfidf %>% right_join(train_ids,by ='index') %>% select(-index)
test_tf     = tfidf %>% anti_join(train_ids,by ='index') %>% select(-index)

fit_tf = rpart(pres_num ~.,training_tf)

tf_train <- predict(fit_tf,type='class')
tf_pred_train   <-  table(training_tf$pres_num,tf_train)

tf_test = predict(fit_tf , newdata = test_tf, type = 'class')
tf_pred_test  = table(test_tf$pres_num,tf_test)

# test accuracy
(tf_test_accuracy  = round(sum(diag(tf_pred_test))/sum(tf_pred_test),3))

save.image(file = "Classification_tree.RData")

                                              # BAG    # ITF
# Balanced data (stop words removed)          0.281    0.281
# Balanced data (stop words not removed)      0.375    0.361

# Unbalanced data ( stop words removed)       0.338    0.338    
# Unbalanced data ( stop words not removed)   0.385    0.373