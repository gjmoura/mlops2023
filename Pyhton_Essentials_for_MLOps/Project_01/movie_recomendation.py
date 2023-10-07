import logging
import re as regex
import pandas as pd
import os
import ipywidgets as widgets
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Constantes
MOVIES_CSV = "./ml-25m/movies.csv"
RATINGS_CSV = "./ml-25m/ratings.csv"

# Funções

def download_data() -> None:
    """
    Faz o download dos datasets.
    """
    url = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"
    zip_filename = "ml-25m.zip"
    extracted_folder_name = "ml-25m"

    try:
        logging.info("Iniciando Download...")

        with requests.Session() as session:
            response = session.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            chunk_size = 128 * 1024
            total_chunks = total_size // chunk_size

            with open(zip_filename, 'wb') as file:
                for data in tqdm(response.iter_content(chunk_size=chunk_size),
                                total=total_chunks,
                                unit='KB',
                                unit_scale=True):
                    file.write(data)
            

        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(".")

        logging.info("Download Finalizado! :)")
        

    except requests.ConnectionError:
        logging.error("Connection error. :(")
    except requests.Timeout:
        logging.error("Timed-out. Try again later. :(")


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Carrega um arquivo csv em um DataFrame

    args:
        filepath: Caminho do arquivo csv.

    return:
        pd.DataFrame: Retorno do DataFrame
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError as error:
        logging.error('File not found: %s', error)
        return pd.DataFrame()


movies_dataset_list = load_dataset(MOVIES_CSV)
ratings_list_of_movies = load_dataset(RATINGS_CSV)

recommendation_list = widgets.Output()


def remove_especial_caracteres_on_string(input_string: str) -> str:
    """
    Remove caracteres especiais

    args:
        input_string: String em que a limpeza será feita.

    return:
        str: String limpa sem caracteres especiais
    """
    cleaned_string = regex.sub("[^a-zA-Z0-9 ]", "", input_string)
    return cleaned_string


def search_movie_by_cosine_similarity(query_text: str) -> pd.DataFrame:
    """
    Busca título pela similidaridade

    args:
        query_text: String em que o cáculo da similaridade será feito.

    return:
        pd.DataFrame: DataFrame com os cálculos da similaridade para cada valor do dataSet
    """
    clean_query_text = remove_especial_caracteres_on_string(query_text)
    query_vectorizer = tfidf_vectorizer.transform([clean_query_text])
    similarity_scores = cosine_similarity(
        query_vectorizer, tfidf_vectorizer_vectors).flatten()
    movies_indice_list = np.argpartition(similarity_scores, -5)[-5:]
    result_movie_list = movies_dataset_list.iloc[movies_indice_list].iloc[::-1]

    return result_movie_list


def find_similar_movies(movie_id: int) -> pd.DataFrame:
    """
    Busca por filmes similares

    args:
        movie_id: Identificador do filme a ser pesquisado.

    return:
        pd.DataFrame: DataFrame com os filmes mais similares na busca
    """
    user_list_with_similar_rating = ratings_list_of_movies[(ratings_list_of_movies["movieId"] == movie_id) & (
        ratings_list_of_movies["rating"] > 4)]["userId"].unique()
    recommendation_list_for_similar_user_rating = ratings_list_of_movies[(ratings_list_of_movies["userId"].isin(
        user_list_with_similar_rating)) & (ratings_list_of_movies["rating"] > 4)]["movieId"]
    recommendation_list_for_similar_user_rating = recommendation_list_for_similar_user_rating.value_counts() / \
        len(user_list_with_similar_rating)

    recommendation_list_for_similar_user_rating = recommendation_list_for_similar_user_rating[
        recommendation_list_for_similar_user_rating > .10]
    all_users_list = ratings_list_of_movies[(ratings_list_of_movies["movieId"].isin(
        recommendation_list_for_similar_user_rating.index)) & (ratings_list_of_movies["rating"] > 4)]
    all_user_recommendation_list = all_users_list["movieId"].value_counts(
    ) / len(all_users_list["userId"].unique())
    recommendation_percentages_dataframe = pd.concat(
        [recommendation_list_for_similar_user_rating, all_user_recommendation_list], axis=1)
    recommendation_percentages_dataframe.columns = ["similar", "all"]

    recommendation_percentages_dataframe["score"] = recommendation_percentages_dataframe["similar"] / \
        recommendation_percentages_dataframe["all"]
    recommendation_percentages_dataframe = recommendation_percentages_dataframe.sort_values(
        "score", ascending=False)
    return recommendation_percentages_dataframe.head(10).merge(movies_dataset_list, left_index=True, right_on="movieId")[["score", "title", "genres"]]


def search_for_similar_movies_based_on_recommendation_and_query_text(data: pd.DataFrame):
    """
    Busca por filmes similares baseados na similaridade

    data:
        pd.DataFrame: DataFrame para busca dos filmes
    """
    with recommendation_list:
        recommendation_list.clear_output()
        query_text = data["new"]
        if len(query_text) >= 4:
            result_list = search_movie_by_cosine_similarity(query_text)
            movie_id = result_list.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))


# Execução do código
movies_dataset_list["clean_title"] = movies_dataset_list["title"].apply(
    remove_especial_caracteres_on_string)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(
    movies_dataset_list["clean_title"])

movie_name_input = widgets.Text(
    value='',
    placeholder='Search for a movie title',
    description='Movie Title:',
    disabled=False
)

movie_name_input.observe(
    search_for_similar_movies_based_on_recommendation_and_query_text, names='value')

display(movie_name_input, recommendation_list)
