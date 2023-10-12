import os
import requests
import xmltodict
import logging

from airflow.decorators import dag, task
import pendulum
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.providers.sqlite.hooks.sqlite import SqliteHook

PODCAST_URL = "https://www.marketplace.org/feed/podcast/marketplace/"
EPISODE_FOLDER = "episodes"

@dag(
    dag_id='podcast_summary',
    schedule_interval="@daily",
    start_date=pendulum.datetime(2022, 5, 30),
    catchup=False,
)
def podcast_summary():
    """
    Initial execution for podcast summary
    """
    create_podcast_database = SqliteOperator(
        task_id='create_table_sqlite',
        sql=r"""
        CREATE TABLE IF NOT EXISTS episodes (
            link TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            published TEXT,
            description TEXT,
            transcript TEXT
        );
        """,
        sqlite_conn_id="podcasts"
    )

    @task()
    def get_podcast_episodes():
        """
        Method to get the list  of podcast episodes

        Return: 
            list: List of podcast episodes
        """
        try:
            data = requests.get(PODCAST_URL)
            feed = xmltodict.parse(data.text)
            episodes: list = feed["rss"]["channel"]["item"]
            print(f"Found {len(episodes)} episodes.")
            return episodes
        except requests.exceptions.HTTPError as request_error:
            logging.error("Error request: %s", request_error)

    podcast_episodes = get_podcast_episodes()
    create_podcast_database.set_downstream(podcast_episodes)

    @task()
    def load_podcast_episodes(episodes: list):
        """
        Load a list podcast episodes
        """
        hook = SqliteHook(sqlite_conn_id="podcasts")
        stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
        new_episodes = []
        for episode in episodes:
            if episode["link"] not in stored_episodes["link"].values:
                filename = f"{episode['link'].split('/')[-1]}.mp3"
                new_episodes.append([episode["link"], episode["title"], episode["pubDate"], episode["description"], filename])

        hook.insert_rows(table='episodes', rows=new_episodes, target_fields=["link", "title", "published", "description", "filename"])

    load_podcast_episodes(podcast_episodes)

    @task()
    def download_podcast_episodes(episodes: list):
        """
        Download the list of podcast episodes
        """
        for episode in episodes:
            name_end = episode["link"].split('/')[-1]
            filename = f"{name_end}.mp3"
            audio_path = os.path.join(EPISODE_FOLDER, filename)
            if not os.path.exists(audio_path):
                print(f"Downloading {filename}")
                audio = requests.get(episode["enclosure"]["@url"])
                with open(audio_path, "wb+") as f:
                    f.write(audio.content)

    download_podcast_episodes(podcast_episodes)

summary = podcast_summary()