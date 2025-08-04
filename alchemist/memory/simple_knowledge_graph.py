"""
Simple Knowledge Graph System for Alchemist Memory (No SQLite dependency).

In-memory storage + FAISS indexing for fast RDF triple retrieval.
Schema v0: (person) -[relation]-> (object)
"""

import json
import numpy as np
import torch
import faiss
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import os


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


class SimpleKnowledgeGraph:
    """
    Simple Knowledge Graph with in-memory storage and FAISS vector indexing.
    
    Schema v0:
    - (person) -[likes]-> (thing)
    - (person) -[dislikes]-> (thing)  
    - (person) -[project]-> (topic)
    - (person) -[health]-> (issue)
    """
    
    def __init__(self, storage_path: str = "simple_kg.pkl", vector_dim: int = 256):
        self.storage_path = storage_path
        self.vector_dim = vector_dim
        
        # In-memory storage
        self.triples: Dict[int, Triple] = {}
        self.next_id = 1
        
        # Indexes for fast lookup
        self.subject_index: Dict[str, List[int]] = {}
        self.predicate_index: Dict[str, List[int]] = {}
        self.object_index: Dict[str, List[int]] = {}
        
        # FAISS index for vector similarity
        self.index = faiss.IndexFlatIP(vector_dim)  # Inner product for cosine similarity
        self.triple_id_to_idx: Dict[int, int] = {}  # Map triple_id to FAISS index
        self.idx_to_triple_id: Dict[int, int] = {}  # Map FAISS index to triple_id
        self.vectors: Dict[int, np.ndarray] = {}  # Store vectors separately
        
        # Load existing data if available
        self.load_from_disk()
        
        print(f"Simple Knowledge Graph initialized:")
        print(f"  Storage: {storage_path}")
        print(f"  Vector dimension: {vector_dim}")
        print(f"  Existing triples: {len(self.triples)}")
    
    def add_triple(self, triple: Triple, vector: Optional[np.ndarray] = None) -> int:
        """
        Add a triple to the knowledge graph.
        
        Args:
            triple: Triple to add
            vector: Optional vector representation
            
        Returns:
            triple_id: ID of inserted triple
        """
        triple_id = self.next_id
        self.next_id += 1
        
        # Store triple
        self.triples[triple_id] = triple
        
        # Update indexes
        self._add_to_index(self.subject_index, triple.subject, triple_id)
        self._add_to_index(self.predicate_index, triple.predicate, triple_id)
        self._add_to_index(self.object_index, triple.object, triple_id)
        
        # Add to FAISS index if vector provided
        if vector is not None:
            # Normalize vector for cosine similarity
            normalized_vector = vector / np.linalg.norm(vector)
            self.index.add(normalized_vector.reshape(1, -1))
            
            # Update mappings
            faiss_idx = self.index.ntotal - 1
            self.triple_id_to_idx[triple_id] = faiss_idx
            self.idx_to_triple_id[faiss_idx] = triple_id
            self.vectors[triple_id] = normalized_vector
        
        return triple_id
    
    def _add_to_index(self, index: Dict[str, List[int]], key: str, triple_id: int):
        """Add triple_id to index under key."""
        if key not in index:
            index[key] = []
        index[key].append(triple_id)
    
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
            if triple_id and triple_id in self.triples:
                triple = self.triples[triple_id]
                results.append((triple, float(sim)))
        
        return results
    
    def search_by_subject(self, subject: str) -> List[Triple]:
        """Search for all triples with given subject."""
        triple_ids = self.subject_index.get(subject, [])
        return [self.triples[tid] for tid in triple_ids if tid in self.triples]
    
    def search_by_predicate(self, predicate: str) -> List[Triple]:
        """Search for all triples with given predicate."""
        triple_ids = self.predicate_index.get(predicate, [])
        return [self.triples[tid] for tid in triple_ids if tid in self.triples]
    
    def search_by_object(self, obj: str) -> List[Triple]:
        """Search for all triples with given object."""
        triple_ids = self.object_index.get(obj, [])
        return [self.triples[tid] for tid in triple_ids if tid in self.triples]
    
    def get_triple_by_id(self, triple_id: int) -> Optional[Triple]:
        """Get triple by ID."""
        return self.triples.get(triple_id)
    
    def count_triples(self) -> int:
        """Count total number of triples."""
        return len(self.triples)
    
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
    
    def save_to_disk(self):
        """Save knowledge graph to disk."""
        data = {
            'triples': self.triples,
            'next_id': self.next_id,
            'subject_index': self.subject_index,
            'predicate_index': self.predicate_index,
            'object_index': self.object_index,
            'triple_id_to_idx': self.triple_id_to_idx,
            'idx_to_triple_id': self.idx_to_triple_id,
            'vectors': self.vectors
        }
        
        with open(self.storage_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Knowledge graph saved to {self.storage_path}")
    
    def load_from_disk(self):
        """Load knowledge graph from disk."""
        if not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'rb') as f:
                data = pickle.load(f)
            
            self.triples = data.get('triples', {})
            self.next_id = data.get('next_id', 1)
            self.subject_index = data.get('subject_index', {})
            self.predicate_index = data.get('predicate_index', {})
            self.object_index = data.get('object_index', {})
            self.triple_id_to_idx = data.get('triple_id_to_idx', {})
            self.idx_to_triple_id = data.get('idx_to_triple_id', {})
            self.vectors = data.get('vectors', {})
            
            # Rebuild FAISS index
            if self.vectors:
                vectors_list = []
                for triple_id in sorted(self.vectors.keys()):
                    vectors_list.append(self.vectors[triple_id])
                
                if vectors_list:
                    vectors_array = np.array(vectors_list)
                    self.index.add(vectors_array)
            
            print(f"Knowledge graph loaded from {self.storage_path}")
            
        except Exception as e:
            print(f"Failed to load knowledge graph: {e}")


def populate_mock_data(kg: SimpleKnowledgeGraph):
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
    kg = SimpleKnowledgeGraph("test_simple_kg.pkl")
    
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
    
    # Save to disk
    kg.save_to_disk()
    print("✅ Knowledge graph test complete!")