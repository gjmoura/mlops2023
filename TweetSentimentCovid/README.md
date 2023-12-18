# 🤖 Projeto Final
Projeto Final da disciplina Projeto de Sistemas Baseados em Aprendizado de Máquina da UFRN, que consiste em desenvolver uma aplicação que incorpora duas ferramentas de Machine Learning Operations (MLOps), sendo elas o Data Version Control (DVC) e o Gradio.

## 📒 Projeto
# Classifying Disaster-Related Tweets as Real or Fake
Esse projeto se chama <i>Análise de sentimentos de Tweets sobre a COVID-19</i> e constrói um modelo de classificação de texto de aprendizagem profunda para prever se tweets expressam emoções neutras, positivas ou negativas relacionadas a pandemia do COVID-19. Utilizando como base um [Projeto](https://www.kaggle.com/code/himanshutripathi/covid-19-tweets-analysis-97-accuracy) e um [conjunto de dados](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/data) disponíveis na plataforma [Kaggle](https://www.kaggle.com/), o algoritmo passa por etapas que incluem a exploração e visualização dos dados, o pré-processamento textual e o treinamento do modelo com TensorFlow. 

## Dependências
- dvc
- Gradio
- Python 3.8+
- numpy
- pandas
- pytest
- requests
- tensorflow
- seaborn
- transformers
- matplotlib
- scikit-learn
- nltk

Instale as dependência do projeto:
```
pip install -r requirements.txt
```

## Como executar
Primeiramente, você precisará se conectar ao repositório remoto do dvc.

```
dvc remote add -d gdrive_remote gdrive://1nkSy4Ho_sjzXhKVb2wLoTaNaHKoxFpve
```

Garantindo a conexão, execute o comando ```dvc pull```, para baixar os data sets utilizados no projeto localmente.

Em seguida é necessário executar o arquivo com o código fonte para treinamento do modelo e execução do ```gradio``` para interação visual com o modelo.

## Resultados

A partir da execução do código foi possível obter dentro do Weights & Biases os histogramas e gráficos que foram passados a partir do Logging. As imagens a seguir mostram, repectivamente, os valores para a quantidade de tweets que estão relacionados a tweets não reais e reais.

![Target_Value](images/Target.png)

A imagem seguinte mostra os valores para cada target descrito, mas dessa vez eles estão normalizados. 

![Normalized_Target_Value](images/Normalized_Target.png)

Já o histograma abaixo realça de forma visual as diferenças entre os targets. 

![histogram](images/histogram_values.png)

Por fim, tem-se a o pipleline da execução das tarefas no AirFlow.

![histogram](images/Airflow.png)

# 📹 Vídeo explicativo
- [Vídeo Loom](https://www.loom.com/share/50f64bc841d0491eac4ecbb3275a57eb)

## ℹ Mais informações

Alunos:
- Adson Emanuel Santos Amaral
- Gustavo Jerônimo Moura de França
- José Augusto Agripino de Oliveira
