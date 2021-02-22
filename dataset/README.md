# Dataset - Hackathon JPA Agro 2021

## Descrição dos arquivos

O arquivo ```dataset_train.csv``` contém as informações para a realização do Hackathon JPA Agro 2021. Ele contém os preços de venda praticados pela JPA Agro para o produto polpa cítrica, no período de janeiro de 2014 até julho de 
2019. O objetivo da competição é criar um preditor dos preços de venda para o período de 30 dias a partir da última data disponível no arquivo ```dataset_train.csv```.

O arquivo ```evalute.py``` obtém a métrica RMSE das predições enviadas. Seu uso se dá da seguinte maneira:

```python3 evaluate.py true_values.txt predicted_values.txt```

Ambos arquivos ```true_values.txt``` e ```predicted_values.txt``` devem conter 30 valores reais separados por vírgula, com arredondamento e duas casas decimais. O arquivo ```true_values.txt``` será disponibilizado publicamente após o envio das soluções.


## Atributos

**product** = produto comercializado (valor constante igual a "Polpa Cítrica")

**negotiation_date** = data do faturamento do produto

**sold_price** = preço de venda do produto (variável a ser predita)


> Você pode usar dados externos que possam ser úteis para a predição, desde que sejam dados públicos. Use
sua criatividade! Enriqueça o banco de dados com outras variáveis que estejam associadas ao mercado de
polpa cítrica.
