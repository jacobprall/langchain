from __future__ import annotations

import enum
import json
import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    import sqlitecloud

logger = logging.getLogger(__name__)

_LANGCHAIN_DEFAULT_TABLE_NAME = "langchain_store"

class SQLiteCloudVec(VectorStore):
    """SQLite Cloud as a vector database.

    To use, you should have a SQLite Cloud account and python packages installed.
    Example:
        .. code-block:: python
            from langchain_community.vectorstores import SQLiteCloudVec
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            ...
    """
    class VectorType(str, enum.Enum):
        """Enumerator for the supported Vector types"""
        FLOAT = "float"
        BIT = "bit"

    def __init__(
        self,
        embedding: Embeddings,
        conn_str: str,
        db_name: str,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME, # type: ignore
        logger: Optional[logging.Logger] = logger,
    ):  
        base_url, params = conn_str.split('?', 1)

        # Insert the database name between the base URL and query parameters
        formatted_conn_str = f"{base_url}/{db_name}?{params}"

        connection = sqlitecloud.connect(formatted_conn_str)

        if not isinstance(embedding, Embeddings):
            warnings.warn("embeddings input must be Embeddings object.")

        self._connection = connection
        self._table_name = table_name
        self._embedding = embedding
        self._logger = logger
        self._vector_type = self.VectorType.FLOAT

        self.create_table_if_not_exists()

    def create_table_if_not_exists(self) -> None:
        self._connection.execute(
            f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS ? USING vec0(
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT,
                    text TEXT,
                    metadata BLOB,
                    text_embedding ?[?]
                );
            """,
            (self._table_name, self._vector_type, self._embedding.get_dimensionality())
        )

    def add_embeddings(
            self,
            texts: Iterable[str],
            embeddings: List[List[float]],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
        ):
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """

        if not metadatas:
            metadatas = [{} for _ in texts]

        if not ids:
            ids = [None for _ in texts]
        
        data_input = [
            (id, text, json.dumps(metadata), json.dumps(embed))
            for id, text, metadata, embed in zip(ids, texts, metadatas, embeddings)
        ]

        self._connection.executemany(
            f"INSERT INTO {self._table_name}(id, text, metadata, text_embedding) VALUES (?,?,?,?)",
            data_input
        )

        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        embeddings = self._embedding.embed_documents(list(texts))
        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def similarity_search(
        self,
        prompt: str,
        k: int = 4,
    ):
        """Run similarity search with SQLite Cloud.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self._embedding.embed_query(prompt)
        # construct query
        query = f"""
            SELECT
                rowid,
                id,
                text,
                metadata,
                distance
            FROM {self._table_name} 
            WHERE text_embedding MATCH {json.dumps(embedding)}
            AND k = {k}
            """
        # execute query
        result = self.connection.execute(query)
        documents = []
        for row in result:
            metadata = json.loads(row["metadata"]) or {}
            doc = Document(page_content=row["text"], metadata=metadata)
            documents.append((doc, row["distance"]))
        return documents
    
    @classmethod
    def from_texts(
        cls: Type[SQLiteCloudVec],
        embedding: Embeddings,
        conn_str: str,
        db_name: str,
        table_name: str,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = logger,
    ) -> SQLiteCloudVec:
        """Return VectorStore initialized from texts and embeddings."""
        
        sqlc_vector = cls(
            embedding=embedding,
            conn_str=conn_str,
            db_name=db_name,
            table_name=table_name,
            logger=logger
        )
        
        sqlc_vector.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return sqlc_vector

    def get_dimensionality(self) -> int:
        """
        Function that does a dummy embedding to figure out how many dimensions
        this embedding function returns. Needed for the virtual table DDL.
        """
        lorem_ipsum = "Lorem Ipsum"
        dummy_embedding = self._embedding.embed_query(lorem_ipsum)
        return len(dummy_embedding)
    
