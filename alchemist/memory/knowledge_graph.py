"""
Knowledge Graph System for Alchemist Memory.

Implements SQLite storage + FAISS indexing for fast RDF triple retrieval.
Schema v0: (person) -[relation]-> (object)
"""

import sqlite3
import json
import numpy as np
import torch
import faiss
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class Triple:
    """RDF Triple: subject -[predicate]-> object"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }
    
    def to_text(self) -> str:
        """Convert triple to natural language text."""
        return f"{self.subject} {self.predicate} {self.object}"


class KnowledgeGraph:
    """
    Knowledge Graph with SQLite storage and FAISS vector indexing.
    
    Schema v0:
    - (person) -[likes]-> (thing)
    - (person) -[dislikes]-> (thing)  
    - (person) -[project]-> (topic)
    - (person) -[health]-> (issue)
    """
    
    def __init__(self, db_path: str = "knowledge_graph.db", vector_dim: int = 256):
        self.db_path = db_path
        self.vector_dim = vector_dim
        
        # Initialize SQLite database
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(vector_dim)  # Inner product for cosine similarity
        self.triple_id_to_idx = {}  # Map triple_id to FAISS index
        self.idx_to_triple_id = {}  # Map FAISS index to triple_id
        
        # Load existing vectors if any
        self.load_existing_vectors()
        
        print(f"Knowledge Graph initialized:")
        print(f"  Database: {db_path}")
        print(f"  Vector dimension: {vector_dim}")
        print(f"  Existing triples: {self.count_triples()}")
    
    def create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()
        
        # Triples table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                timestamp TEXT,
                vector_blob BLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Index for fast lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predicate ON triples(predicate)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_object ON triples(object)')
        
        self.conn.commit()
    
    def add_triple(self, triple: Triple, vector: Optional[np.ndarray] = None) -> int:
        """
        Add a triple to the knowledge graph.
        
        Args:
            triple: Triple to add
            vector: Optional vector representation
            
        Returns:
            triple_id: ID of inserted triple
        """
        cursor = self.conn.cursor()
        
        # Convert vector to blob if provided
        vector_blob = vector.tobytes() if vector is not None else None
        
        cursor.execute('''
            INSERT INTO triples (subject, predicate, object, confidence, timestamp, vector_blob)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (triple.subject, triple.predicate, triple.object, triple.confidence, triple.timestamp, vector_blob))
        
        triple_id = cursor.lastrowid
        self.conn.commit()
        
        # Add to FAISS index if vector provided
        if vector is not None:
            # Normalize vector for cosine similarity
            normalized_vector = vector / np.linalg.norm(vector)
            self.index.add(normalized_vector.reshape(1, -1))
            
            # Update mappings
            faiss_idx = self.index.ntotal - 1
            self.triple_id_to_idx[triple_id] = faiss_idx
            self.idx_to_triple_id[faiss_idx] = triple_id
        
        return triple_id
    
    def search_by_vector(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Triple, float]]:
        """
        Search for similar triples using vector similarity.
        
        Args:
            query_vector: Query vector [vector_dim]
            k: Number of results to return
            
        Returns:
            List of (triple, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query vector
        normalized_query = query_vector / np.linalg.norm(query_vector)
        
        # Search FAISS index
        similarities, indices = self.index.search(normalized_query.reshape(1, -1), min(k, self.index.ntotal))
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            triple_id = self.idx_to_triple_id.get(idx)
            if triple_id:
                triple = self.get_triple_by_id(triple_id)
                if triple:
                    results.append((triple, float(sim)))
        
        return results
    
    def search_by_subject(self, subject: str) -> List[Triple]:
        """Search for all triples with given subject."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM triples WHERE subject = ?', (subject,))
        
        results = []
        for row in cursor.fetchall():
            triple = self._row_to_triple(row)
            results.append(triple)
        
        return results
    
    def search_by_predicate(self, predicate: str) -> List[Triple]:
        """Search for all triples with given predicate."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM triples WHERE predicate = ?', (predicate,))
        
        results = []
        for row in cursor.fetchall():
            triple = self._row_to_triple(row)
            results.append(triple)
        
        return results
    
    def get_triple_by_id(self, triple_id: int) -> Optional[Triple]:
        """Get triple by ID."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM triples WHERE id = ?', (triple_id,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_triple(row)
        return None
    
    def _row_to_triple(self, row) -> Triple:
        """Convert database row to Triple object."""
        return Triple(
            subject=row[1],
            predicate=row[2],
            object=row[3],
            confidence=row[4],
            timestamp=row[5]
        )
    
    def count_triples(self) -> int:
        """Count total number of triples."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM triples')
        return cursor.fetchone()[0]
    
    def load_existing_vectors(self):
        """Load existing vectors into FAISS index."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, vector_blob FROM triples WHERE vector_blob IS NOT NULL')
        
        vectors = []
        triple_ids = []
        
        for row in cursor.fetchall():
            triple_id, vector_blob = row
            if vector_blob:
                vector = np.frombuffer(vector_blob, dtype=np.float32)
                if len(vector) == self.vector_dim:
                    # Normalize vector
                    normalized_vector = vector / np.linalg.norm(vector)
                    vectors.append(normalized_vector)
                    triple_ids.append(triple_id)
        
        if vectors:
            vectors_array = np.array(vectors)
            self.index.add(vectors_array)
            
            # Update mappings
            for i, triple_id in enumerate(triple_ids):
                self.triple_id_to_idx[triple_id] = i
                self.idx_to_triple_id[i] = triple_id
    
    def create_persona_vector(self, user: str, memory_specialist, tokenizer) -> np.ndarray:
        """
        Create a persona vector for a user by combining all their triples.
        
        Args:
            user: Username
            memory_specialist: Memory specialist model  
            tokenizer: Tokenizer
            
        Returns:
            persona_vector: [vector_dim] aggregated representation
        """
        user_triples = self.search_by_subject(user)
        
        if not user_triples:
            # Return zero vector if no information
            return np.zeros(self.vector_dim, dtype=np.float32)
        
        # Convert triples to text and encode
        vectors = []
        for triple in user_triples:
            text = triple.to_text()
            json_data = {"user": user, "text": text}
            
            with torch.no_grad():
                vector = memory_specialist.encode_json_input(json_data, tokenizer)
                vectors.append(vector.cpu().numpy().flatten())
        
        # Mean pooling
        if vectors:
            persona_vector = np.mean(vectors, axis=0)
            return persona_vector.astype(np.float32)
        else:
            return np.zeros(self.vector_dim, dtype=np.float32)
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def populate_mock_data(kg: KnowledgeGraph):
    """Populate knowledge graph with 50-100 mock triples for testing."""
    
    mock_triples = [
        # Jeff's preferences
        Triple("Jeff", "likes", "ukulele"),
        Triple("Jeff", "likes", "coffee"),
        Triple("Jeff", "likes", "cats"),
        Triple("Jeff", "dislikes", "math"),
        Triple("Jeff", "project", "alchemist-ai"),
        Triple("Jeff", "health", "insomnia"),
        Triple("Jeff", "owns", "cat named Whiskers"),
        Triple("Jeff", "drinks", "espresso"),
        Triple("Jeff", "plays", "ukulele daily"),
        Triple("Jeff", "works on", "neural networks"),
        
        # Sarah's preferences  
        Triple("Sarah", "likes", "painting"),
        Triple("Sarah", "likes", "tea"),
        Triple("Sarah", "dislikes", "loud music"),
        Triple("Sarah", "project", "art-gallery"),
        Triple("Sarah", "health", "allergies"),
        Triple("Sarah", "owns", "dog named Max"),
        Triple("Sarah", "drinks", "green tea"),
        Triple("Sarah", "paints", "landscapes"),
        Triple("Sarah", "studies", "art history"),
        
        # Alex's preferences
        Triple("Alex", "likes", "gaming"),
        Triple("Alex", "likes", "pizza"),
        Triple("Alex", "dislikes", "vegetables"),
        Triple("Alex", "project", "game-engine"),
        Triple("Alex", "health", "nearsighted"),
        Triple("Alex", "owns", "gaming PC"),
        Triple("Alex", "plays", "strategy games"),
        Triple("Alex", "codes", "C++"),
        Triple("Alex", "streams", "on Twitch"),
        
        # Maria's preferences
        Triple("Maria", "likes", "books"),
        Triple("Maria", "likes", "wine"),
        Triple("Maria", "dislikes", "crowds"),
        Triple("Maria", "project", "novel-writing"),
        Triple("Maria", "health", "migraines"),
        Triple("Maria", "owns", "library of 500 books"),
        Triple("Maria", "reads", "mystery novels"),
        Triple("Maria", "writes", "fiction"),
        Triple("Maria", "speaks", "three languages"),
        
        # Relationships and interactions
        Triple("Jeff", "friends with", "Sarah"),
        Triple("Sarah", "collaborates with", "Maria"),
        Triple("Alex", "roommate of", "Jeff"),
        Triple("Maria", "mentor to", "Sarah"),
        
        # More detailed preferences
        Triple("Jeff", "favorite genre", "indie folk"),
        Triple("Jeff", "morning routine", "coffee and ukulele"),
        Triple("Jeff", "weekend activity", "hiking"),
        Triple("Sarah", "favorite color", "blue"),
        Triple("Sarah", "inspiration", "Van Gogh"),
        Triple("Sarah", "studio location", "downtown loft"),
        Triple("Alex", "favorite game", "Civilization VI"),
        Triple("Alex", "setup", "dual monitor"),
        Triple("Alex", "streaming schedule", "evenings"),
        Triple("Maria", "writing time", "early morning"),
        Triple("Maria", "favorite author", "Agatha Christie"),
        Triple("Maria", "current project", "detective novel"),
        
        # Skills and abilities
        Triple("Jeff", "skill", "guitar playing"),
        Triple("Jeff", "skill", "machine learning"),
        Triple("Sarah", "skill", "watercolor painting"),
        Triple("Sarah", "skill", "art curation"),
        Triple("Alex", "skill", "game development"),
        Triple("Alex", "skill", "live streaming"),
        Triple("Maria", "skill", "creative writing"),
        Triple("Maria", "skill", "literary analysis"),
        
        # Goals and aspirations
        Triple("Jeff", "goal", "release ukulele album"),
        Triple("Jeff", "goal", "improve AI systems"),
        Triple("Sarah", "goal", "solo art exhibition"),
        Triple("Sarah", "goal", "art therapy certification"),
        Triple("Alex", "goal", "indie game success"),
        Triple("Alex", "goal", "1000 Twitch followers"),
        Triple("Maria", "goal", "publish novel"),
        Triple("Maria", "goal", "teach creative writing"),
    ]
    
    print(f"Adding {len(mock_triples)} mock triples to knowledge graph...")
    
    for triple in mock_triples:
        kg.add_triple(triple)
    
    print(f"Mock data populated! Total triples: {kg.count_triples()}")


if __name__ == "__main__":
    # Test the knowledge graph
    kg = KnowledgeGraph("test_kg.db")
    
    # Populate with mock data
    populate_mock_data(kg)
    
    # Test searches
    print("\n🔍 Testing searches:")
    jeff_triples = kg.search_by_subject("Jeff")
    print(f"Jeff's triples: {len(jeff_triples)}")
    for triple in jeff_triples[:3]:
        print(f"  {triple.to_text()}")
    
    likes_triples = kg.search_by_predicate("likes")
    print(f"'Likes' triples: {len(likes_triples)}")
    for triple in likes_triples[:3]:
        print(f"  {triple.to_text()}")
    
    kg.close()