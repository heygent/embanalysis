from functools import cached_property
from pathlib import Path

import duckdb
import pandas as pd


from embanalysis.sampler import EmbeddingsSampleMeta
from embanalysis.constants import DB_PATH


class DuckDBLoader:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)

    @cached_property
    def conn(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(self.db_path)

    def init_db(self):
        self.conn.query("""
            CREATE OR REPLACE TABLE embeddings (
                model_id VARCHAR NOT NULL,
                token_id INTEGER NOT NULL,
                token VARCHAR NOT NULL,
                embeddings FLOAT[] NOT NULL,
                PRIMARY KEY (model_id, token_id),
            );

            CREATE OR REPLACE SEQUENCE embedding_id_seq;
            CREATE OR REPLACE TABLE samples (
                sample_id INTEGER DEFAULT NEXTVAL('embedding_id_seq') PRIMARY KEY,
                model_id VARCHAR NOT NULL,
                meta JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            );

            CREATE OR REPLACE TABLE embedding_to_sample (
                model_id VARCHAR NOT NULL,
                token_id INTEGER NOT NULL,
                sample_id INTEGER NOT NULL,
                PRIMARY KEY (model_id, token_id, sample_id),
            );
        """)

    def drop_db(self):
        self.conn.close()
        if self.db_path.exists():
            self.db_path.unlink()
        del self.conn

    def store_sample(self, sample: pd.DataFrame, meta: EmbeddingsSampleMeta):
        model_id = sample["model_id"].iloc[0]

        sample_id = self.conn.execute(
            """
            INSERT INTO samples (model_id, meta)
            VALUES (?, ?)
            RETURNING sample_id;
        """,
            (model_id, meta),
        ).fetchone()[0]  # pyright: ignore

        self.conn.execute("""
            INSERT INTO embeddings (model_id, token_id, token, embeddings)
            SELECT * FROM sample
            ON CONFLICT DO NOTHING;
        """)

        sample["sample_id"] = sample_id

        self.conn.execute("""
            INSERT INTO embedding_to_sample (model_id, token_id, sample_id)
            SELECT model_id, token_id, sample_id
            FROM sample;
        """)

    def list_samples(self) -> pd.DataFrame:
        """List all available samples in the database."""
        query = """
            SELECT samples.sample_id, samples.model_id, meta, created_at, COUNT(*) as sample_size
            FROM samples
            JOIN embedding_to_sample ON samples.sample_id = embedding_to_sample.sample_id
            GROUP BY samples.sample_id, samples.model_id, meta, created_at
            ORDER BY created_at DESC;
        """

        return self.conn.execute(query).fetchdf()

    def list_models(self) -> list[str]:
        return self.conn.execute("SELECT DISTINCT model_id FROM samples").fetchall()

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            del self.conn

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
