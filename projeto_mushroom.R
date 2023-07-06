# Importação das bibliotecas utilizadas
suppressPackageStartupMessages(for (package in c('caret','readr','ggplot2',
                                                 'dplyr','corrplot','rpart',
                                                 'rpart.plot','randomForest',
                                                 'utils','e1071','nnet')) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package)
    library(package,character.only=T)
  }
})
# ============================================================================
# Download do dataset
# URL do arquivo zip
url <- "https://archive.ics.uci.edu/static/public/73/mushroom.zip"

# Nome do arquivo zip
nome_arquivo_zip <- "mushroom.zip"

# Nome do diretório de extração
diretorio_extracao <- "dados"

# Download do arquivo zip
download.file(url, destfile = nome_arquivo_zip)

# Criar diretório para a extração
dir.create(diretorio_extracao)

# Extrair o arquivo zip
unzip(nome_arquivo_zip, exdir = diretorio_extracao)

# Atribuir dataset ao objeto mushroom
mushroom <- read.csv("dados/agaricus-lepiota.data", header = FALSE, sep = ",",
                     stringsAsFactors = T)

# Visualização do Dataset
View(mushroom)
str(mushroom)
# ============================================================================
# Pré-processamento dos dados
# Alterando o nome das colunas
colnames(mushroom) <- c('target', 'cap.shape', 'cap.surface', 'cap.color',
                        'bruises', 'odor', 'gill.attachment', 'gill.spacing',
                        'gill.size', 'gill.color', 'stalk.shape',
                        'stalk.root', 'stalk.surface.above.ring',
                        'stalk.surface.below.ring','stalk.color.above.ring',
                        'stalk.color.below.ring', 'veil.type', 'veil.color',
                        'ring.number', 'ring.type', 'spore.print.color',
                        'population', 'habitat')

# mushroom <- as.factor(mushroom)
levels(mushroom$target) <- c("edible", "poisonous")

str(mushroom)
# ============================================================================
# EDA
mushroom_table <- lapply(seq(from=2, to=ncol(mushroom)), 
                         function(x) {table(mushroom$target, mushroom[,x])})
names(mushroom_table) <- colnames(mushroom)[2:ncol(mushroom)]
for(i in 1:length(mushroom_table)) {
  print("======================================")
  print(names(mushroom_table)[i])
  print(mushroom_table[[i]]) 
}

# Removendo a coluna porque tem apenas um atributo na categoria
mushroom <- mushroom[- which(colnames(mushroom) == "veil.type")]

# Verificando valores ausentes
valores_ausentes <- is.na(mushroom)
colSums(valores_ausentes)
sum(colSums(valores_ausentes))

# Distribuição do target
table(mushroom$target)

# ===========================================================================
# Convertendo as variáveis categóricas para numéricas
mushroom <- mushroom %>%
  mutate(across(everything(), as.numeric))

# Verificando a estrutura do dataframe após a transformação
str(mushroom)
View(mushroom)

# ===========================================================================
# Separação do dataset em treino e test com o target desbalanceado
train_p <- 0.8
set.seed(42)
index <- createDataPartition(y = mushroom$target, p = train_p, list = FALSE, times = 1)
train_data <- mushroom[index, ]
test_data <- mushroom[-index, ]

X_train <- train_data[, -ncol(train_data)]
y_train <- train_data$target
X_test <- test_data[, -ncol(test_data)]
y_test <- test_data$target

# ===========================================================================
# Controle dos parâmetros para o treino
# fitControl <- trainControl(method = "cv", number = 10)

# ===========================================================================
# Funções úteis
# Visualiza os gráficos importantes
plotROC <- function(actual_data, predicted_data) {
  pROC_obj <- roc(as.integer(actual_data), as.integer(predicted_data),
                  smoothed = TRUE,  direction="<",
                  # arguments for ci
                  ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                  # arguments for plot
                  plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                  print.auc=TRUE, show.thres=TRUE)
  sens.ci <- ci.se(pROC_obj)
  plot(sens.ci, type="shape", col="lightblue")
  plot(sens.ci, type="bars")
}

# Dados dos modelos importantes
modelprint <- function(model, test_data) {
  print("=========================================")
  print(model)
  print("=========================================")
  print(varImp(model))
  print("=========================================")
  print(confusionMatrix(data = predict(model, newdata = test_data), test_data$target, positive='poisonous'))
  print("=========================================")
  plotROC(test_data$target, predict(model, test_data))
}

# ============================================================================
# Redes MLP
train_data$target = as.factor(train_data$target)


nn <- nnet(target ~ ., data = train_data, size = 5, maxit = 500)
# Calcular a acurária do treino
preds_train = predict(nn,train_data,type = "class")
# Calcular a acurácia do teste
preds_teste = predict(nn,test_data,type = "class")
# Train
matrix_conf = table(preds_train, train_data$target)

acertos = diag(as.matrix(matrix_conf))

acc_train = sum(acertos)/length(preds_train)
# Test
matrix_conf = table(preds_teste, test_data$target)

acertos = diag(as.matrix(matrix_conf))

acc_test = sum(acertos)/length(preds_teste)

# ============================================================================
# SVM
model_svm <- svm(target ~.,data=train_data, cost=10,gamma = 0.0001,type = "C-classification")

# Calcular a acurária do treino
preds_train_svm <- predict(model_svm, train_data)
# Calcular a acurária do teste
preds_test_svm = predict(model_svm,test_data)

# Train
matrix_conf_train_svm <- table(preds_train_svm, train_data$target)
acertos_svm_train <- diag(as.matrix(matrix_conf_train_svm))
acc_train_svm <- sum(acertos_svm_train)/length(preds_train_svm)

# Test
matrix_conf_test_svm = table(preds_test_svm, test_data$target)
acertos_svm_test = diag(as.matrix(matrix_conf_test_svm))
acc_test_svm = sum(acertos_svm_test)/length(preds_test_svm)
# ============================================================================
# Interpretação dos modelos
# 