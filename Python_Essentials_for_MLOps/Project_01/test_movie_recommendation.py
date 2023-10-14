import os
from movie_recomendation import remove_especial_caracteres_on_string
from movie_recomendation import download_data


def test_remove_special_caracteres_from_string() -> None:
    """
    Testa remoção de caracteres especiais
    """
    assert remove_especial_caracteres_on_string("Teste!@") == "Teste"
    assert remove_especial_caracteres_on_string(
        "Star Wars: Threads of Destiny") == "Star Wars Threads of Destiny"

def test_download_data() -> None:
    """
    Test the download of data.

    Returns:
        None
    """
    
    download_data()
    # Check if the file exists
    assert os.path.isfile("./ml-25m.zip")
