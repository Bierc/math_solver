# Projeto: Treinamento de CNN para Reconhecimento de Números e Símbolos Matemáticos

## Descrição

Este projeto tem como objetivo desenvolver um solucionador de equações matemáticas em tempo real, utilizando um modelo de Redes Neurais Convolucionais (CNN) treinado para o reconhecimento de números e operadores matemáticos manuscritos.

## Objetivo

O objetivo deste estudo é desenvolver um sistema capaz de reconhecer e resolver equações matemáticas em tempo real. Para isso, foi desenvolvido um modelo baseado em CNNs treinado para identificar números de 0 a 9 e operadores matemáticos básicos como `+`, `-`, `*`, e `=`.

## Fundamentação Teórica

O projeto foi dividido em duas etapas principais:

1. **Treinamento do Modelo**: Utilizando bibliotecas como TensorFlow e OpenCV, o modelo foi treinado para reconhecer números e operadores matemáticos manuscritos. Diversas técnicas de processamento de imagem, como binarização, detecção de contornos e padding, foram aplicadas para melhorar a precisão do modelo.
   
2. **Implementação em Tempo Real**: Após o treinamento, o modelo foi implementado em um sistema que captura imagens em tempo real, processa essas imagens para detectar e resolver equações matemáticas.

### Geração de Equações

Para treinar o modelo, foi gerada uma base de dados de equações matemáticas utilizando imagens de números e operadores manuscritos. Estas equações incluem números de 0 a 9 e operadores como soma, subtração e multiplicação. O conjunto de dados gerado é utilizado para avaliar o desempenho do modelo.

## Processamento em Tempo Real

O processamento em tempo real de equações matemáticas é realizado através de várias etapas, cada uma desempenhando um papel crucial para garantir a precisão do reconhecimento e solução da equação.

## Código do Projeto

O código deste projeto está dividido em várias partes que abrangem desde a geração de equações até a criação de imagens de equações manuscritas para o treinamento do modelo.

### Principais Pastas e Scripts

- data
    - Contém as pastas com as imagens dos numeros e simbolos utilizados para o treinamento
- models
    - Contém os modelos gerados após o treinamento
- src
    - scripts
        - Códigos para execução do modelo em tempo real
    - training
        - Códigos para treinamento do modelo

1. [Geração de Equações Lineares e Aritméticas](src\training\equation_generator.ipynb)

2. [Treinamento do modelo](src\training\train_classifier.ipynb)

3. [Script para reconhecimento das equações em tempo real](src\scripts\realtime_equation_solver.py)

O código completo pode ser encontrado no arquivo principal do repositório e está documentado com explicações sobre cada função e sua utilização.
