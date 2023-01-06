# Definindo o diretório de trabalho
setwd("C:/Users/marcos/Documents/Cientista_de_Dados/BigDataRAzure/Projetos_com_Feedback/Arquivos/Projeto_2")
getwd()


# Carregando Pacotes
library(farff)
library(ggplot2)
library(tidyverse)
library(randomForest)
library(C50)
library(e1071)
library(class)
library(caret)
library(caTools)

# Carregando os Dados
df <- readARFF("Acoustic_Extinguisher_Fire_Dataset.arff")


# Visualizando os Dados
View(df)
head(df)


# Informações sobre o Dados
str(df)
summary(df)

unique(df$SIZE)
unique(df$FUEL)
unique(df$DISTANCE)
unique(df$CLASS)


# Etapa 3: #### Análise Exploratória dos Dados - Limpeza dos Dados #####

# Extraindo os nomes das colunas, e gravando em um vetor
colnames(df)
new_names <- colnames(df)  

# Renomendo os nomes do vetor acima
new_names[1] <- "Tamanho"
new_names[2] <- "Tipo_Combustivel"
new_names[3] <- "Distancia"
new_names[4] <- "Decibeis"
new_names[5] <- "Fluxo_Ar"
new_names[6] <- "Frequencia"
new_names[7] <- "Extincao_Chama"

# Renomendo as colunas do Data Frame
colnames(df) <- new_names
View(df)


# Removendo os objetos anteriores para liberar memória RAM
rm(new_names)


# Função para verificar se existe valores ausêntes
verifica_na <- function(x){
  colSums(is.na(x))
}


# Chamando a função "verifica_na"
verifica_na(df)

# Observações: Não existem valores NA nesse conjunto de dado
# Convertendo o tipo da variável "Tamanho" de "numeric" para o tipo "factor"
# O tamnho do tanques estão divididos em 7 categorias.
df$Tamanho <- as.factor(df$Tamanho)
str(df$Tamanho)


# Alterando o label da variável target, "Extincao_Chama"
df$Extincao_Chama = sapply(df$Extincao_Chama, function(x){ifelse(x==0, 'NAO', 'SIM')})
View(df)
str(df$Extincao_Chama)


# Convertendo a variável em factor, quando o label foi renomeado o tipo foi alterado para caracter
df$Extincao_Chama <- as.factor(df$Extincao_Chama)
str(df$Extincao_Chama)


# Verificando a proporção das Observações na variável preditora
table(df$Extincao_Chama)
proporcao <- round(prop.table(table(df$Extincao_Chama)) * 100, digits = 1)
proporcao

# Gráfico da operação acima
summary(df$Extincao_Chama)

ggplot(df, aes(x = Extincao_Chama)) +
  geom_bar() + 
  scale_y_continuous(limits = c(0,9000), breaks = seq(0,9000,500)) + 
  ggtitle("Contagem Variável Extincao_Chama") + 
  theme(legend.position = "right")


#### Conclusão da análise acima: ####

# Pode-se verificar que a variável "Extincao_Chama está balanceada. 


##### Plots e Estatísticas ##### 



# Modelo randomForest para criar um plot de importância das variáveis
modelo <- randomForest( Extincao_Chama ~ .,data = df, ntree = 100, nodesize = 10, importance = T)

varImpPlot(modelo)

#### Conclusão da análise acima: ####

# A seleção de váriaveis nos dois parâmetros "Accuracy" e o "Gini", foram diferentes. Para a 
# primeiro modelo com o RandomForest vamos selecionar as 3 primeiras variáveis escolhidas pelo 
# "Gine", no segundo modelo com o RandomForest vou treinar o modelo com todas variáveis e comparar
# os dois modelos.


##### Análise Preditiva ##### 

# Divisão dos dados em Treino e Teste

set.seed(458)
linhas <- sample(1:nrow(df), 0.7 * nrow(df))

dados_treino <- df[linhas,]
dim(dados_treino)
View(dados_treino)
head(dados_treino)


dados_teste <- df[-linhas,]
dim(dados_teste)
View(dados_teste)
head(dados_teste)


#### Modelos com RandomForest ####


# Criando o Modelo Preditivo com RandomForest, versão 1:
modelo_rf_v1 <- randomForest( Extincao_Chama ~ Fluxo_Ar
                        + Distancia
                        + Frequencia,
                        data = dados_treino, 
                        ntree = 100, 
                        nodesize = 10)
modelo_rf_v1 # Error rate: 11.18%


# Previsão 1 nos dados de teste:
previsao_rf_v1 <- predict(modelo_rf_v1, dados_teste)
previsao_rf_v1
mean(previsao_rf_v1 == dados_teste$Extincao_Chama) # Percentual de acerto: 89.22%


# Estatísticas da Previsão acima:
confusionMatrix(dados_teste$Extincao_Chama, previsao_rf_v1, positive = 'SIM') # Accuracy: 89.22%
roc.curve(dados_teste$Extincao_Chama, previsao_rf_v1, plotit = T, col = "red") # ACU: 0.89


# Criando o Modelo Preditivo com RandomForest, versão 2:
modelo_rf_v2 <- randomForest(Extincao_Chama ~ ., data = dados_treino, ntree = 100, nodesize = 10)
modelo_rf_v2 # Error rate: 3.99%


# Previsão 2 nos dados de teste:
previsao_rf_v2 <- predict(modelo_rf_v2, dados_teste)
previsao_rf_v2
mean(previsao_rf_v2 == dados_teste$Extincao_Chama) # Percentual de acerto: 96.73% 


# Estatísticas da Previsão acima: 
confusionMatrix(dados_teste$Extincao_Chama, previsao_rf_v2, positive = 'SIM') # Accuracy: 96.73%
roc.curve(dados_teste$Extincao_Chama, previsao_rf_v2, plotit = T, col = "red") # AUC: 0.967


#### Modelo com C5.0 ####


# Criando o Modelo Preditivo C5.0, versão 1:
modelo_c50_v1 <- C5.0(Extincao_Chama ~., data = dados_treino)
modelo_c50_v1


# Previsão  nos dados de teste:
previsao_c50_v1 <- predict(modelo_c50_v1, dados_teste)
previsao_c50_v1
mean(previsao_c50_v1 == dados_teste$Extincao_Chama) # Percentual de acerto: 96.35%


# Estatísticas da Previsão acima: 
confusionMatrix(dados_teste$Extincao_Chama, previsao_c50_v1, positive = 'SIM') # Accuracy: 96.35%
roc.curve(dados_teste$Extincao_Chama, previsao_c50_v1, plotit = T, col = "red") # AUC: 0.963


#### Modelo com SVM ####


# Criando o Modelo Preditivo SVM, versão 1:
modelo_svm_v1 <- svm(Extincao_Chama ~ ., 
                     data = dados_treino, 
                     type = 'C-classification', 
                     kernel = 'radial')
modelo_svm_v1


# Previsões nos dados de teste
previsao_svm_v1 <- predict(modelo_svm_v1, dados_teste) 
mean(previsao_svm_v1 == dados_teste$Extincao_Chama) # Percentual de acerto: 94.85%


# Estatísticas da Previsão acima: 
confusionMatrix(dados_teste$Extincao_Chama, previsao_svm_v1, positive = 'SIM') # Accuracy: 94.86%
roc.curve(dados_teste$Extincao_Chama, previsao_svm_v1, plotit = T, col = "red") # AUC: 0.949


#### Conclusão dos Modelos Criados acima: ####
# Fora criados 4 modelos, com 3 algoritimos diferentes, 2 modelos com o RandomForest, onde no 
# primeiro modelo 1, foi utilizado 3 variáveis de entrada e um segundo modelo com todas as variáveis de entrada
# do conjunto de dados, foi criado 1 modelo com c5.0 e 1 modelo com o algoritimo SVM. Os modelos
# "modelo_rf_v2" e o modelo "modelo_c50_v1" apresentaram o nével de acuracia bem próximos.
# Como Modelo Final, vou escolher o segundo modelo criado com o RandomFores, "modelo_rf_v2".


