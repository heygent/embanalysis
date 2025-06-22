from functools import cached_property
from pathlib import Path

import duckdb
import pandas as pd


from embanalysis.sampler import EmbeddingsSampleMeta
from embanalysis.constants import DB_PATH


class DuckDBLoader:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self._init_db()
    
    @cached_property
    def conn(self):
        return duckdb.connect(self.db_path)
    
    def _init_db(self):
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
        
    def store_sample(self, sample: pd.DataFrame, meta: EmbeddingsSampleMeta) -> pd.DataFrame:
        model_id = sample['model_id'].iloc[0]

        sample_id = self.conn.execute("""
            INSERT INTO samples (model_id, meta)
            VALUES (?, ?)
            RETURNING sample_id;
        """, (model_id, meta)).fetchone()[0]

        self.conn.execute("""
            INSERT INTO embeddings (model_id, token_id, token, embeddings)
            SELECT * FROM sample
        """)

        sample['sample_id'] = sample_id

        self.conn.execute("""
            INSERT INTO embedding_to_sample (model_id, token_id, sample_id)
            SELECT model_id, token_id, sample_id
            FROM (VALUES (?, ?, ?)) AS v(model_id, token_id, sample_id);
        """, sample[['model_id', 'token_id', 'sample_id']].values.tolist())
    
    def list_available_embeddings(self) -> pd.DataFrame:
        """List all available embeddings in the database."""
        query = """
            SELECT model_id, embedding_type, metadata, COUNT(*) as num_embeddings, 
                   MIN(created_at) as created_at
            FROM embeddings
            GROUP BY model_id, embedding_type, metadata
            ORDER BY model_id, embedding_type
        """
        
        result = self.conn.execute(query).fetchdf()
        return result
    
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.close()


def main():
    import sys

    with DuckDBLoader("embeddings.duckdb") as loader:
        if len(sys.argv) > 1:
            model_id = sys.argv[1]
        else:
            exit("Usage: python duckdb_loader.py [model_id]")
        



if __name__ == "__main__":
    
    loader = DuckDBLoader("embeddings.duckdb")
