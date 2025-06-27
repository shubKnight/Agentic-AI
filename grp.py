from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional, Dict, List, Union
class Neo4jTM:
    def __init__(self, uri: str, user: str, password: str, embedding_model: Optional[str] = 'all-MiniLM-L6-v2'):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_model = None
        if embedding_model:
            self.embedding_model = SentenceTransformer(embedding_model)
    
    def close(self) -> None:
        self._driver.close()

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        if self.embedding_model and text:
            return self.embedding_model.encode(text).tolist()
        return None
    
    def _filter_embeddings(self, properties: Dict) -> Dict:
        
        return {k: v for k, v in properties.items() if not k.endswith('_embedding')}
    
    def create_node(self, label: str, properties: Optional[Dict] = None) -> str:

        properties = properties or {}
        properties_copy = properties.copy()

        if self.embedding_model:
             for key, value in properties.items(): 
                 if isinstance(value, str) and key not in ['date', 'deadline']:
                    embedding = self._generate_embedding(value)
                    if embedding:
                        properties_copy[f"{key}_embedding"] = embedding  # Add to copy

        with self._driver.session() as session:
            result = session.execute_write(self._create_node, label, properties_copy)
            return result
   
    @staticmethod
    def _create_node(tx, label, properties) -> str:
        query = f"CREATE (n:{label} $props) RETURN elementId(n) as node_id"
        result = tx.run(query, props=properties)
        return result.single()["node_id"]
    
    def create_relationship(self, from_id: str, to_id: str, rel_type: str, properties: Optional[Dict] = None) -> str:
       
        with self._driver.session() as session:
            result = session.execute_write(self._create_relationship, from_id, to_id, rel_type, properties or {})
            return result

    @staticmethod
    def _create_relationship(tx, from_id, to_id, rel_type, properties):
        query = (
            "MATCH (a), (b) "
            "WHERE elementId(a) = $from_id AND elementId(b) = $to_id "
            f"CREATE (a)-[r:{rel_type} $props]->(b) "
            "RETURN elementId(r) as rel_id"
        )
        result = tx.run(query, from_id=from_id, to_id=to_id, props=properties)
        return result.single()["rel_id"]
    
    def delete_node(self, node_id: str) -> bool:
        with self._driver.session() as session:
            result = session.execute_write(self._delete_node, node_id)
            return result
        
    @staticmethod
    def _delete_node(tx, node_id: str) -> bool:
        query = (
            "MATCH (n) "
            "WHERE elementId(n) = $node_id "
            "DETACH DELETE n "
            "RETURN count(n) > 0 as deleted"
        )
        result = tx.run(query, node_id=node_id)
        return result.single()["deleted"]
    
    def delete_relationship(self, rel_id: str) -> bool:
       
        with self._driver.session() as session:
            result = session.execute_write(self._delete_relationship, rel_id)
            return result
        
    @staticmethod
    def _delete_relationship(tx, rel_id: str) -> bool:
        query = (
            "MATCH ()-[r]->() "
            "WHERE elementId(r) = $rel_id "
            "DELETE r "
            "RETURN count(r) > 0 as deleted"
        )
        result = tx.run(query, rel_id=rel_id)
        return result.single()["deleted"]
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        
        with self._driver.session() as session:
            result = session.execute_read(self._get_node, node_id)
            if result:
                return self._filter_embeddings(result)
            return None
        
        
    @staticmethod
    def _get_node(tx, node_id: str) -> Optional[Dict]:
        query = (
            "MATCH (n) "
            "WHERE elementId(n) = $node_id "
            "RETURN properties(n) as props"
        )
        result = tx.run(query, node_id=node_id)
        record = result.single()
        return record["props"] if record else None

    def find_similar_nodes(self, text: str, label: str, top_k: int = 5) -> List[Dict]:
        
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
            
        embedding = self._generate_embedding(text)
        
        with self._driver.session() as session:
            result = session.execute_read(self._find_similar_nodes, label, embedding, top_k)
            return [{"id": record["id"],"properties": self._filter_embeddings(record["props"]),"similarity": record["similarity"] } for record in result]
    
    @staticmethod
    def _find_similar_nodes(tx, label: str, embedding: List[float], top_k: int) -> List[Dict]:
        query = (
            f"MATCH (n:{label}) "
            "WHERE n.name_embedding IS NOT NULL "
            "WITH n, vector.similarity.cosine($embedding, n.name_embedding) AS similarity "
            "ORDER BY similarity DESC "
            "LIMIT $top_k "
            "RETURN elementId(n) as id, properties(n) as props, similarity"
        )
        result = tx.run(query, embedding=embedding, top_k=top_k)
        return [dict(record) for record in result]
    
    def get_team_members(self, project_id: str) -> List[Dict]:
        
        with self._driver.session() as session:
            result = session.execute_read(self._get_team_members, project_id)
            return [{"person_id": record["person_id"],"person": self._filter_embeddings(record["person_props"]),"role": record["relationship_props"]} for record in result]
        
    @staticmethod
    def _get_team_members(tx, project_id: str) -> List[Dict]:
        query = (
            "MATCH (p:Project)<-[r:WORKING_ON]-(person:Person) "
            "WHERE elementId(p) = $project_id "
            "RETURN elementId(person) as person_id, properties(person) as person_props, "
            "properties(r) as relationship_props"
        )
        result = tx.run(query, project_id=project_id)
        return [dict(record) for record in result]
    
    def get_projects_by_person(self, person_id: str) -> List[Dict]:
       
        with self._driver.session() as session:
            result = session.execute_read(self._get_projects_by_person, person_id)
            return [{"project_id": record["project_id"],"project": self._filter_embeddings(record["project_props"]),"involvement": record["relationship_props"]} for record in result]
        
    @staticmethod
    def _get_projects_by_person(tx, person_id: str) -> List[Dict]:
        query = (
            "MATCH (person:Person)-[r:WORKING_ON]->(p:Project) "
            "WHERE elementId(person) = $person_id "
            "RETURN elementId(p) as project_id, properties(p) as project_props, "
            "properties(r) as relationship_props"
        )
        result = tx.run(query, person_id=person_id)
        return [dict(record) for record in result]
    
    def update_node_properties(self, node_id: str, new_properties: Dict) -> bool:
       
        with self._driver.session() as session:
            result = session.execute_write(self._update_node_properties, node_id, new_properties)
            return result
        
    @staticmethod
    def _update_node_properties(tx, node_id: str, new_properties: Dict) -> bool:
        query = (
            "MATCH (n) "
            "WHERE elementId(n) = $node_id "
            "SET n += $props "
            "RETURN count(n) > 0 as updated"
        )
        result = tx.run(query, node_id=node_id, props=new_properties)
        return result.single()["updated"]
    
    def add_skill_to_person(self, person_id: str, skill_name: str, proficiency: str = "Intermediate") -> int:
        
        with self._driver.session() as session:
            result = session.execute_write(self._add_skill_to_person, person_id, skill_name, proficiency)
            return result
        
    @staticmethod
    def _add_skill_to_person(tx, person_id: str, skill_name: str, proficiency: str) -> int:
        query = (
            "MATCH (person:Person) WHERE elementId(person) = $person_id "
            "MERGE (skill:Skill {name: $skill_name}) "
            "MERGE (person)-[r:HAS_SKILL]->(skill) "
            "SET r.proficiency = $proficiency "
            "RETURN elementId(r) as rel_id"
        )
        result = tx.run(query, person_id=person_id, skill_name=skill_name, proficiency=proficiency)
        return result.single()["rel_id"]
    
    def recommend_team_for_project(self, project_id: str, required_skills: List[str]) -> List[Dict]:
        
        with self._driver.session() as session:
            result = session.execute_read(self._recommend_team_for_project, project_id, required_skills)
            return [{"person_id": record["person_id"],"person": self._filter_embeddings(record["person_props"]),"matching_skills": record["matching_skills"],"match_count": record["match_count"]} for record in result]
        
    @staticmethod
    def _recommend_team_for_project(tx, project_id: str, required_skills: List[str]) -> List[Dict]:
        query = (
            "MATCH (p:Project) WHERE elementId(p) = $project_id "
            "MATCH (person:Person)-[r:HAS_SKILL]->(skill:Skill) "
            "WHERE skill.name IN $required_skills "
            "WITH person, collect(skill.name) as matching_skills "
            "RETURN elementId(person) as person_id, properties(person) as person_props, "
            "matching_skills, size(matching_skills) as match_count "
            "ORDER BY match_count DESC"
        )
        result = tx.run(query, project_id=project_id, required_skills=required_skills)
        return [dict(record) for record in result]
    
if __name__ == "__main__":
    manager = Neo4jTM("neo4j+s://b31ce023.databases.neo4j.io", "neo4j", "npNo7WUVA3TLi4pMsLdI5qIMNKuxVeTpOu1YpEx0x7A",embedding_model="all-MiniLM-L6-v2")
    
    try:
        # Create team members
        mihir_id = manager.create_node("Person", {"name": "Mihir", "role": "Developer"})
        bhavya_id = manager.create_node("Person", {"name": "Bhavya", "role": "Designer"})
        shubham_id = manager.create_node("Person", {"name": "Shubham", "role": "Developer"})
        krrish_id=manager.create_node("Person", {"name": "Krrish", "role": "Designer"})
        
        # Create projects
        ai_project_id = manager.create_node("Project", {"name": "Agentic AI Platform","description": "Building next-generation AI agents","deadline": "2025-06-20"})
        
        web_project_id = manager.create_node("Project", {"name": "Website Redesign","description": "Modernizing the team website","deadline": "2025-07-15"})
        
        # Create relationships
        manager.create_relationship(mihir_id, ai_project_id, "WORKING_ON", {"role": "Lead Developer","since": "2025-05-15" })
        
        manager.create_relationship(bhavya_id, ai_project_id, "WORKING_ON", {"role": "UI/UX Designer","since": "2025-05-01"})
        
        manager.create_relationship(shubham_id, ai_project_id, "MANAGING", {"since": "2025-05-01"})
        
        manager.create_relationship(krrish_id, ai_project_id, "WORKING_ON", {"role": "Face Recognition Developer","since": "2025-05-18"})
        
        # Add skills to people
        manager.add_skill_to_person(mihir_id, "Python", "Expert")
        manager.add_skill_to_person(mihir_id, "Machine Learning", "Advanced")
        manager.add_skill_to_person(bhavya_id, "UI Design", "Expert")
        manager.add_skill_to_person(shubham_id, "Project Management", "Expert")
        manager.add_skill_to_person(krrish_id, "Face Recognition", "Expert")

        
        # Query examples
        print("\nTeam members for AI project:")
        print(manager.get_team_members(ai_project_id))
        
        print("\nProjects Mihir is working on:")
        print(manager.get_projects_by_person(mihir_id))
        
        print("\nRecommend team for AI project needing Python and Machine Learning:")
        print(manager.recommend_team_for_project(ai_project_id, ["Python", "Machine Learning"]))
        
        print("\nFind similar projects to 'AI system':")
        print(manager.find_similar_nodes("AI system", "Project"))
        
    finally:
        manager.close()
