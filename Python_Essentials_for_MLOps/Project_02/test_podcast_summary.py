from podcast_summary import get_podcast_episodes

# Sample data for testing
MOCK_EPISODES = [
    {
        "link": "https://episode01.com/episode01",
        "title": "Episode 01",
        "pubDate": "2023-10-15",
        "description": "Description",
        "enclosure": {"@url": "https://episode01.com/episode01.mp3"}
    },
    {
        "link": "https://episode02.com/episode02",
        "title": "Episode 02",
        "pubDate": "2023-10-15",
        "description": "Description",
        "enclosure": {"@url": "https://episode02.com/episode02.mp3"}
    },
    {
        "link": "https://episode03.com/episode03",
        "title": "Episode 03",
        "pubDate": "2023-10-15",
        "description": "Description",
        "enclosure": {"@url": "https://episode03.com/episode03.mp3"}
    },
]

def test_get_podcast_episodes():
  """
  Tests the method to get podcast episodes
  """

  episodes = get_podcast_episodes()

  # Check the result
  assert len(episodes) == 3
  assert episodes[0]['title'] == 'Episode 01'
