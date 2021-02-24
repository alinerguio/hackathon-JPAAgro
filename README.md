# Hackathon JPAAgro

## Desafio

As especificações do desafio podem ser vistas nesse [repositório](https://github.com/dsrg-icet/hackathon_JPAAgro/tree/main/dataset).

## Sobre a solução

O melhor modelo encontrado para esta solução, foi uma LSTM com apenas uma camada oculta de 350 neurônios. Para o treinamento, foi utilizado o otimizador Adam com uma taxa de aprendizado fixada em 0,001 e 1.000 repetições. Para a melhoria na inicialização dos pesos, foi utilizada uma técnica de normalização. Os batches de treinamento foram sendo obtidos aleatoriamente a cada passo do treinamento.

## Disposição de arquivos

### code

Códigos de aprendizado de máquina.

### dataset

Dados disponibilizados para o desafio.

### envio

Arquivos enviados como resposta do desafio, contendo:
    - [Relatório](https://github.com/alinerguio/hackathon-JPAAgro/blob/main/envio/template_paper.pdf) descrevendo as etapas que a equipe executou;
    - [Código](https://github.com/alinerguio/hackathon-JPAAgro/blob/main/envio/main.py) em sua versão final;
    - [Predição](https://github.com/alinerguio/hackathon-JPAAgro/blob/main/envio/predicted_values.txt) necessária para o desafio.

### requirements.txt

Bibliotecas necessárias para execução 

