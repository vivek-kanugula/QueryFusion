from hybrid_rag.settings import Settings


def test_heroku_postgres_url_normalization():
    # Heroku legacy postgres:// URL format
    heroku_url = "postgres://user123:secretpass@ec2-54-123.compute-1.amazonaws.com:5432/d123456"
    settings = Settings(database_url=heroku_url)
    assert settings.database_url == "postgresql+psycopg://user123:secretpass@ec2-54-123.compute-1.amazonaws.com:5432/d123456"


def test_standard_postgresql_url_normalization():
    std_url = "postgresql://user123:secretpass@localhost:5432/testdb"
    settings = Settings(database_url=std_url)
    assert settings.database_url == "postgresql+psycopg://user123:secretpass@localhost:5432/testdb"


def test_psycopg_url_preserved():
    psycopg_url = "postgresql+psycopg://user123:secretpass@localhost:5432/testdb"
    settings = Settings(database_url=psycopg_url)
    assert settings.database_url == psycopg_url


def test_database_url_direct_normalization():
    direct_url = "postgres://direct_user:pass@direct_host:5432/direct_db"
    settings = Settings(database_url_direct=direct_url)
    assert settings.database_url_direct == "postgresql+psycopg://direct_user:pass@direct_host:5432/direct_db"

