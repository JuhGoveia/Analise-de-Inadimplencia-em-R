# Carregar bibliotecas
library(dplyr)
library(readr)
library(ggplot2)
library(caret)
library(Metrics)
library(glmnet)
library(randomForest)
library(xgboost)
library(rpart)

# Função de pré-processamento
preprocess_data <- function(df) {
  colnames(df) <- make.names(colnames(df), unique = TRUE)
  categorical_vars <- names(df)[sapply(df, is.character)]
  for (char_col in categorical_vars) {
    df[[char_col]] <- as.numeric(as.factor(df[[char_col]]))
  }
  return(df)
}

# Carregar dados de treinamento
train_data <- read_csv("C:/Users/julia/OneDrive/Documentos/Documents/Faculdade/train.csv")
head(train_data)

# Pré-processar os dados de treinamento
train_data <- preprocess_data(train_data)

# Calcular a matriz de correlação
correlation_matrix <- cor(train_data)
target_correlations <- sort(correlation_matrix['target', ], decreasing = TRUE)
top_correlations <- c(head(target_correlations, 11), tail(target_correlations, 10))
principal_correlations <- names(top_correlations)

# Selecionar recursos com base nas principais correlações
train_features <- train_data[, principal_correlations]

# Separar os dados em treinamento e validação
X <- train_features %>% select(-target)
y <- train_features$target
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE, times = 1)
X_train <- X[trainIndex, ]
X_val <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_val <- y[-trainIndex]

# Criar conjuntos de treinamento e validação
train_set <- cbind(X_train, target = y_train)
validation_set <- cbind(X_val, target = y_val)

# Inicializar os modelos
modelo_linear <- lm(target ~ ., data = train_set)
ridge_model <- glmnet(as.matrix(train_set[, -ncol(train_set)]), train_set$target, alpha = 0)
lasso_model <- glmnet(as.matrix(train_set[, -ncol(train_set)]), train_set$target, alpha = 1)
tree_model <- rpart(target ~ ., data = train_set)

# Usar xgboost para Gradient Boosting
dtrain <- xgb.DMatrix(data = as.matrix(train_set[, -ncol(train_set)]), label = train_set$target)
gradiente_boosting_model <- xgboost(data = dtrain, objective = "reg:squarederror", nrounds = 100)

# Lista de modelos para iterar
modelos <- list(
  "Regressão Linear" = modelo_linear,
  "Regressão de Ridge" = ridge_model,
  "Regressão de Lasso" = lasso_model,
  "Árvore de Decisão" = tree_model,
  "Gradiente Boosting" = gradiente_boosting_model
)

# Lista para armazenar os RMSE de cada modelo
model_rmse <- list()

# Treinar e avaliar cada modelo
for (nome in names(modelos)) {
  if (nome %in% c("Regressão de Ridge", "Regressão de Lasso")) {
    y_val_pred <- predict(modelos[[nome]], newx = as.matrix(validation_set[, -ncol(validation_set)]), s = 0.01)
  } else if (nome == "Gradiente Boosting") {
    dval <- xgb.DMatrix(data = as.matrix(validation_set[, -ncol(validation_set)]))
    y_val_pred <- predict(modelos[[nome]], newdata = dval)
  } else {
    y_val_pred <- predict(modelos[[nome]], newdata = validation_set)
  }
  
  rmse <- rmse(validation_set$target, y_val_pred)
  model_rmse[[nome]] <- rmse
}

model_rmse

# Carregar dados de teste
test_data <- read_csv("teste.csv")

# Pré-processar os dados de teste
test_data <- preprocess_data(test_data)
recursos_teste <- test_data[, principal_correlations[-1]]

# Preparar os dados de teste para fazer previsões
X_teste = recursos_teste
dtest <- xgb.DMatrix(data = as.matrix(X_teste))

# Fazer previsões com o modelo de Gradient Boosting
test_predictions <- predict(gradiente_boosting_model, newdata = dtest)

# Adicionar as previsões ao conjunto de teste
test_data <- cbind(test_data, target = test_predictions)
head(test_data$target)

# Treinar um modelo Random Forest
random_forest_model <- randomForest(target ~ ., data = train_set[, -ncol(train_set)])

# Fazer previsões usando o modelo Random Forest nos dados de teste
test_predictions_rf <- predict(random_forest_model, newdata = test_data)

# Adicionar as previsões do Random Forest ao conjunto de teste
test_data <- cbind(test_data, target_rf = test_predictions_rf)
head(test_data$target_rf)

