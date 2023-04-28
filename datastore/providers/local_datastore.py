import json
import os
import heapq
import numpy as np
from typing import Dict, List, Optional
from datastore.datastore import DataStore
from models.models import (
    DocumentChunk,
    DocumentMetadataFilter,
    DocumentChunkWithScore,
    DocumentMetadataFilter,
    QueryResult,
    QueryWithEmbedding,
)

class LocalDataStore(DataStore):
    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    async def delete(self, 
                     ids: Optional[List[str]] = None,
                     filter: Optional[DocumentMetadataFilter] = None,
                     delete_all: Optional[bool] = None) -> bool:
        """
        Removes documents by ids, filter, or everything in the datastore.
        Returns whether the operation was successful.
        """
        # Delete all documents from the index if delete_all is True
        if delete_all:
            try:
                for file_name in os.listdir(self.storage_path):
                    file_path = os.path.join(self.storage_path, file_name)
                    os.remove(file_path)
                return True
            except Exception as e:
                raise(e)

        # Delete by filter
        if filter:
            if filter.document_id:
                try:
                    file_path = os.path.join(self.storage_path, f"{filter.document_id}.json")
                    os.remove(file_path)
                    return True
                except Exception as e:
                    raise(e)

        # Delete by explicit ids
        if ids:
            try:
                for doc_id in ids:
                    file_path = os.path.join(self.storage_path, f"{doc_id}.json")
                    os.remove(file_path)
                return True
            except Exception as e:
                raise(e)

        # If no ids or filter specified, raise exception
        raise ValueError("No filter or ids specified for delete operation")

    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """
        Takes in a list of list of document chunks and inserts them into the local storage.
        Return a list of document ids.
        """
        # Initialize a list of ids to return
        doc_ids: List[str] = []

        # Loop through the dict items
        for doc_id, chunk_list in chunks.items():

            # Append the id to the ids list
            doc_ids.append(doc_id)

            # Serialize all chunks associated with a document and write to local storage
            serialized_chunks = []
            for i, chunk in enumerate(chunk_list):
                serialized_chunk = self._get_chunk_data(chunk)
                serialized_chunks.append(serialized_chunk)

            # Write serialized chunks to local storage
            file_path = os.path.join(self.storage_path, f'{doc_id}.json')
            with open(file_path, 'w') as f:
                json.dump(serialized_chunks, f)

        return doc_ids
    
    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and
        returns a list of query results with matching document chunks and scores.
        """
        # Prepare results object
        results: List[QueryResult] = []

        # Iterate through queries
        for query in queries:
            # Extract query embedding
            query_embedding = np.array(query.embedding, dtype=np.float64)
            query_topk = query.top_k
            # Search for matching documents
            query_results: List[DocumentChunkWithScore] = []
            for file_name in os.listdir(self.storage_path):
                file_path = os.path.join(self.storage_path, file_name)
                with open(file_path, "r") as f:
                    document = json.load(f)
                for doc_chunk in document:
                    # Extract chunk embedding
                    chunk_embedding = np.array(doc_chunk["embedding"])

                    # Calculate cosine similarity between query and document embeddings
                    similarity = np.dot(query_embedding, chunk_embedding) \
                    / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
                    # similarity = cosine_similarity([query_embedding], [document_embedding])[0][0]

                    chunk = DocumentChunkWithScore(
                        id=doc_chunk["metadata"]["document_id"],
                        score=similarity,
                        text=doc_chunk["text"],
                        metadata=doc_chunk["metadata"],
                    )
                    heapq.heappush(query_results, (similarity, chunk))
                    if len(query_results) >= query_topk:
                        heapq.heappop(query_results)
            query_results = [x[1] for x in query_results]
            query_results = sorted(query_results, key=lambda x: x.score, reverse=True)
            # Add query results to overall results
            results.append(QueryResult(query=query.query, results=query_results))

        return results
    
    def _get_chunk_data(self, chunk: DocumentChunk) -> dict:
        """
        Convert DocumentChunk into a JSON object for local storage

        Args:
            chunk (DocumentChunk): Chunk of a Document.

        Returns:
            dict: JSON object for storage in Redis.
        """
        # Convert chunk -> dict
        data = chunk.__dict__
        metadata = chunk.metadata.__dict__
        data["chunk_id"] = data.pop("id")

        # Prep Redis Metadata
        local_metadata = dict()
        if metadata:
            for field, value in metadata.items():
                if value:
                    if field == "created_at":
                        pass
                        # local_metadata[field] = to_unix_timestamp(value)  # type: ignore
                    else:
                        local_metadata[field] = value
        data["metadata"] = local_metadata
        return data