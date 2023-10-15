# Airflow Data Pipeline to Download Podcasts
Esse é projeto se chama <i>Build an Airflow Data Pipeline to Download Podcasts</i> e nele foi construído um pipeline de dados de quatro etapas usando o Airflow, que é uma ferramenta popular de engenharia de dados baseada em Python para definir e executar pipelines de dados muito poderosos e flexíveis. O pipeline baixará episódios de podcast. Armazenaremos nossos resultados em um banco de dados SQLite que podemos consultar facilmente.. É um projeto de protifólio disponível na plataforma [Dataquest](https://app.dataquest.io/)

## Dependências
- Airflow 2.7.1
- Python 3.8+
- astroid
- numpy
- pandas
- pylint
- pytest
- pydub
- requests
- sqlite3
- vosk
- xmltodict


## Instruções

1. Clone o projeto localmente: 
   ```
   git clone https://github.com/gjmoura/mlops2023.git
   ```
2. Acesse a pasta `Project 02` dentro do diretório `Python_Essentials_for_MLOps`.
3. Instale as dependências necessárias:
   ```bash
    python --version
    
    CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-2.3.1/constraints-3.9.txt"
    
    pip install "apache-airflow==2.3.1" --constraint "${CONSTRAINT_URL}"
    ```
    ```
    pip install -r requirements.txt
    ```
    
4. Rode airflow server no terminal
    ```
    airflow standalone
    ```
  
5. Criação do banco de dados
    * Rode no terminal `sqlite3 episodes.db`
    * Digite `.databases` no prompt para criar o banco de dados
    * Rode `airflow connections add 'podcasts' --conn-type 'sqlite' --conn-host 'episodes.db'`
    *  `airflow connections get podcasts` para ver informações sobre a conexão

6. Execução do código via terminal ou utilizando uma IDE:
    * `mkdir episodes` para criar a pasta com os podcasts
    ```
    python podcast_summary.py -t
    ```
    * Você também pode acompanhar o funcionamento na interface do Airflow
7. Para executar os testes rode o comando:
   ```
    pytest 
   ```
8. Para rodar o pylint no código rode o comando:
   ```
    pylint podcast_summary.py
   ```
