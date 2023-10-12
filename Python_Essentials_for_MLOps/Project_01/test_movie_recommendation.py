# Implementasr testes unitários]
from movie_recomendation import remove_especial_caracteres_on_string


def test_remove_special_caracteres_from_string() -> None:
    """
    Testa remoção de caracteres especiais
    """
    assert remove_especial_caracteres_on_string("Teste!@") == "Teste"
    assert remove_especial_caracteres_on_string(
        "Star Wars: Threads of Destiny") == "Star Wars Threads of Destiny"
