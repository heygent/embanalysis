from dataclasses import asdict
from pathlib import Path
import json

import duckdb
import pandas as pd


from embanalysis.sample_data import (
    EmbeddingsSample,
    EmbeddingsSampleMeta,
    make_meta_object,
)
from embanalysis.constants import DB_PATH


class DuckDBLoader:
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    @classmethod
    def from_path(cls, db_path: str | Path = DB_PATH, read_only: bool = False):
        """Create a DuckDBLoader instance from a given path."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(db_path, read_only=read_only)
        return cls(conn)

    @classmethod
    def default(cls, *args, **kwargs):
        return cls.from_path(*args, **kwargs)

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

    def store_sample(
        self, model_id: str, embeddings_df: pd.DataFrame, meta: EmbeddingsSampleMeta
    ):
        sample_id = self.conn.execute(
            """
            INSERT INTO samples (model_id, meta)
            VALUES (?, ?)
            RETURNING sample_id;
        """,
            (model_id, asdict(meta)),
        ).fetchone()[0]  # pyright: ignore

        self.conn.execute(
            """
            INSERT INTO embeddings (model_id, token_id, token, embeddings)
            SELECT ?, * FROM embeddings_df
            ON CONFLICT DO NOTHING;
        """,
            (model_id,),
        )

        self.conn.execute(
            """
            INSERT INTO embedding_to_sample (model_id, token_id, sample_id)
            SELECT ?, token_id, ?
            FROM embeddings_df;
        """,
            (model_id, sample_id),
        )

    def get_model_samples(self, model_id: str) -> dict[str, EmbeddingsSample]:
        """Get all samples for a specific model."""
        query = """
            SELECT samples.sample_id, samples.meta
            FROM samples
            WHERE samples.model_id = ?;
        """
        sample_ids_and_meta = self.conn.execute(query, (model_id,)).fetchall()
        samples = {}
        for sample_id, meta in sample_ids_and_meta:
            query = """
                SELECT embeddings.token_id, token, embeddings
                FROM embeddings, embedding_to_sample
                WHERE embeddings.token_id = embedding_to_sample.token_id
                AND embeddings.model_id = ?
                AND sample_id = ?;
            """
            embeddings_df = self.conn.execute(query, (model_id, sample_id)).fetchdf()
            meta = make_meta_object(json.loads(meta))
            samples[meta.tag] = EmbeddingsSample(
                meta=meta,
                embeddings_df=embeddings_df,
                sample_id=sample_id,
            )
        return samples

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
        result = self.conn.execute("SELECT DISTINCT model_id FROM samples").fetchall()
        return [row[0] for row in result]

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            del self.conn

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
