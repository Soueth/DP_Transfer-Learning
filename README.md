# Desafio de Projeto - Fazendo o primeiro Transfer-Learning 💿

### O desafio consiste em, a partir de uma base montada por si, treinar um modelo em duas situações: do zero e a partir de um modelo pré-treinado, VGG no caso do desafio. A partir dos resultados, são verificados os benefícios do tranfer-learning.

## Base utilizada

A base utilizada consiste de duas classes: cartas de Yu-gi-oh e cartas de Pokemon.

## Resultados

Os resultados, com 10 épocas, também se divergem. Enquanto no modelo treinado do zero a acurácia dos testes fica um pouco acima de 70%, utilizando-se ao invés o modelo VGG, com os pesos 'imagenet', chega a 80%. Já quando utiliza-se 20 épocas a diferença se torna mas gritante. Enquanto o que não foi utilizado pesos ou um modelo pré-treinado gira em torno de 65%, o que foi treinado usando esse artifício beira a 90%.

## Colab

Link do código no Colab: https://colab.research.google.com/drive/1YNvQ152ImEpEHj-IR4WaQtWFqXJUJog9#scrollTo=6hKYWy0Lm2Ew

Dar upload de dataset.zip e depois utilizar o código indicado para extrair o arquivo.
