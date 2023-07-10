# Importação das bibliotecas utilizadas
suppressPackageStartupMessages(for (package in c('caret','readr','ggplot2',
                                                 'dplyr','corrplot','rpart',
                                                 'rpart.plot','randomForest',
                                                 'utils','e1071','nnet', 'pROC',
                                                 'naiveBayes')) {
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
# mushroom <- mushroom %>%
#   mutate(across(everything(), as.numeric))

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

# ============================================================================
# Redes MLP
train_data$target <- as.factor(train_data$target)
test_data$target <- as.factor(test_data$target)

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
# Distribuição dos targets
plyr::count(mushroom[ ,1])

# carregando o conjunto de dados "mushroom" e convertendo a coluna "class" para numérica
mushroom.data <- mushroom
mushroom.data$target <- as.numeric(mushroom.data$target) - 1
head(mushroom.data$target)

# Divisão do train e test
set.seed(42)
mushroom.train = mushroom%>%
  sample_frac(0.6)
mushroom.test = mushroom%>%
  setdiff(mushroom.train)

# Seleção das features mais importantes usando RFE
rfe_classifier <- rfe(x=mushroom.train[, -1], y=mushroom.train$target,
                      size=c(5, 10, 15, 20, 25), 
                      rfeControl = rfeControl(functions = rfFuncs))
rfe_classifier

# Features 
rfe_classifier$optVariables

# Gráfico do número de variáveis
ggplot(rfe_classifier)

# Modelo SVM com 5 melhor variáveis
set.seed(1)
svm.fit =tune(svm, target ~ odor + spore.print.color + gill.size + gill.color + 
                population,
              data = mushroom.train, 
              kernel = "radial", 
              ranges =list(cost =c(0.1, 1, 5, 10, 100),
                           gamma = c(0.5,1,2,3,4))
)

summary(svm.fit)

# Executando o modelo com os melhores parâmetros
svm.best = svm.fit$best.model
summary(svm.best)

# Predição
svm.pred <- predict(svm.best, newdata=mushroom.test)
#svm.pred <- ifelse(svm.pred > 0.5, 'p', 'e')

# Matriz de confusão
confusionMatrix(data = as.factor(svm.pred), reference = mushroom.test$target)

#Área sob a Curva ROC
library("pROC")

svm.pred.train = predict(svm.best, mushroom.train, decision.values=TRUE)

roc(mushroom.train[, 1], as.numeric(as.factor(svm.pred.train)), plot=TRUE, print.auc = TRUE, legacy.axes=TRUE, main='ROC-Training Predication (SVM)')

roc(mushroom.test[, 1], as.numeric(as.factor(svm.pred)), plot=TRUE, print.auc = TRUE, legacy.axes=TRUE, main='ROC-Testing Predication (SVM)')

# Comparação com Naive Bayes
nb.fit = naiveBayes(as.factor(target)~odor + spore.print.color + gill.size + gill.color + population + 
                      stalk.root + habitat + stalk.surface.above.ring + cap.color + ring.type, 
                    data=mushroom.train)
nb.fit

nb.pred <- predict(nb.fit, mushroom.test)

confusionMatrix(data = as.factor(nb.pred), reference = as.factor(mushroom.test$target))

nb.pred.train = predict(nb.fit, mushroom.train, decision.values=TRUE)

roc(mushroom.train[, 1], as.numeric(as.factor(nb.pred.train)), plot=TRUE, print.auc = TRUE, legacy.axes=TRUE, main='ROC-Training Predication (NB)')

roc(mushroom.test[, 1], as.numeric(as.factor(nb.pred)), plot=TRUE, print.auc = TRUE, legacy.axes=TRUE, main='ROC-Testing Predication (NB)')


# ============================================================================
# Interpretação dos modelos
# Features importantes e global