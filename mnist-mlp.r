# Trains a simple deep neural network on the MNIST dataset.

library(keras)
#install_tensorflow()
library(tensorflow)

# Data Preparation ---------------------------------------------------

batch_size <- 128
num_classes <- 10
epochs <- 3

# load mnist dataset, also splitting into train and test 
mnist <- dataset_mnist() 
x_train <- mnist$train$x 
y_train <- mnist$train$y 
x_test <- mnist$test$x
y_test <- mnist$test$y

dim(x_train)
dim(y_train)
dim(x_test)
dim(y_test)

# formally it was 60000 28 28 for x_train 
#array_reshape(x, dim) with x as an array to be reshaped with the new dim dimension
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
# with dimension 60,000 x 10 , each column for each class and the correct label (class)
# to each observation (each row) will be assigned 1 in that particular class (that column)


y_test <- to_categorical(y_test, num_classes)
# analogously with y_train


# Define Model --------------------------------------------------------------

# units: dimensionality of the output space.
# Dimensionality of the input (integer) not including the samples axis. 
    # This argument is required when using this layer as the first layer in a model.

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Training & Evaluation ----------------------------------------------------

# Fit model to data
history <- model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_split = 0.2
)

plot(history)

score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)

# Output metrics
cat('Test loss:', score[[1]], '\n')
cat('Test accuracy:', score[[2]], '\n')