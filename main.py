import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
import json
from typing import List, Dict, Any
class LawyerMatchingSystem:
    def __init__(self):
        """Initialize the LawyerMatchingSystem with necessary components"""
        self.skills_data = None
        self.lawyer_data = None
        self.skill_vectors = None
        self.lawyer_map = {}  # Map between skills responses and lawyer profiles
        self.client = anthropic.Client(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
    def load_data(self) -> None:
        """Load and preprocess the skills and lawyer data"""
        # Load skills data
        self.skills_data = pd.read_csv('skills_responses.csv')
        self.lawyer_data = pd.read_csv('BD_Caravel.csv')
        
        # Create mapping between skills responses and lawyer profiles using email
        self.lawyer_map = {
            email: idx for idx, email in enumerate(self.skills_data['Submitter Email'])
        }
        
        # Extract skill columns (those containing 'Skill' in the name)
        self.skill_columns = [col for col in self.skills_data.columns if '(Skill' in col]
        
        # Normalize skill scores
        scaler = MinMaxScaler()
        self.skills_data[self.skill_columns] = scaler.fit_transform(
            self.skills_data[self.skill_columns].fillna(0)
        )
        
        # Create skill vectors for matching
        self.skill_vectors = self.skills_data[self.skill_columns].to_numpy()
        
        # Clean lawyer data
        self.lawyer_data = self.lawyer_data.fillna('')
        
    def get_claude_analysis(self, query: str) -> Dict[str, Any]:
        """Use Claude to analyze the legal need and extract relevant skills"""
        prompt = f"""Based on our legal skill taxonomy, analyze this legal need: "{query}"

        Our skill taxonomy includes these main areas:
        {', '.join(self.skill_columns[:10])}... and more.

        Return a JSON object with:
        1. "primary_skills": List of 3-5 most relevant skills from our taxonomy
        2. "required_experience": Minimum years recommended (integer)
        3. "industry_focus": List of relevant industries
        4. "practice_areas": List of general practice areas needed
        
        Return only the JSON object, no other text."""
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return json.loads(response.content[0].text)
    
    def create_skill_vector(self, required_skills: List[str]) -> np.ndarray:
        """Convert required skills into a weighted vector"""
        skill_vector = np.zeros(len(self.skill_columns))
        for skill in required_skills:
            # Find closest matching skill in our taxonomy
            matches = [col for col in self.skill_columns if skill.lower() in col.lower()]
            for match in matches:
                idx = self.skill_columns.index(match)
                skill_vector[idx] = 1.0
        return skill_vector
    
    def match_lawyers(self, requirements: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
        """Match lawyers based on requirements and return top matches"""
        # Create skill vector from requirements
        required_skills_vector = self.create_skill_vector(requirements['primary_skills'])
        
        # Calculate similarity scores
        similarities = cosine_similarity([required_skills_vector], self.skill_vectors)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        matches = []
        for idx in top_indices:
            # Get corresponding lawyer profile
            lawyer_email = self.skills_data.iloc[idx]['Submitter Email']
            lawyer_matches = self.lawyer_data[
                self.lawyer_data['Email'].str.lower() == lawyer_email.lower()
            ]
            
            if len(lawyer_matches) == 0:
                continue
                
            lawyer_profile = lawyer_matches.iloc[0]
            
            # Calculate match percentage
            match_score = similarities[idx] * 100
            
            # Create match info
            match_info = {
                'name': f"{lawyer_profile['First Name']} {lawyer_profile['Last Name']}",
                'title': lawyer_profile['Level/Title'],
                'practice_areas': lawyer_profile['Area of Practise + Add Info'],
                'industry_experience': lawyer_profile['Industry Experience'],
                'languages': lawyer_profile['Languages'],
                'location': lawyer_profile['Location'],
                'match_score': match_score,
                'relevant_skills': [
                    skill for skill, score in zip(self.skill_columns, self.skill_vectors[idx])
                    if score > 0.5  # Only include skills with high scores
                ]
            }
            matches.append(match_info)
            
        return matches

def main():
    st.set_page_config(
        page_title="Rolodex AI - Caravel Law",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Rolodex AI for Caravel Law")
    
    # Initialize the matching system
    matcher = LawyerMatchingSystem()
    
    try:
        matcher.load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
        
    # User input section
    with st.container():
        st.subheader("🔍 Find Your Perfect Legal Match")
        legal_need = st.text_area(
            "Please describe your legal requirements in detail:",
            height=150,
            help="The more specific you are, the better we can match you with the right attorney."
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            search_button = st.button("🔍 Find Matching Attorneys", type="primary")
    
    # Process matching when button is clicked
    if search_button and legal_need:
        with st.spinner("Analyzing your requirements..."):
            try:
                # Get requirements analysis
                requirements = matcher.get_claude_analysis(legal_need)
                
                # Find matches
                matches = matcher.match_lawyers(requirements)
                
                # Display analysis
                st.subheader("📊 Requirements Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Primary Skills Needed:**")
                    for skill in requirements['primary_skills']:
                        st.write(f"- {skill}")
                        
                with col2:
                    st.write("**Industry Focus:**")
                    for industry in requirements['industry_focus']:
                        st.write(f"- {industry}")
                
                # Display matches
                st.subheader("👥 Top Matching Attorneys")
                
                for match in matches:
                    with st.expander(
                        f"{match['name']} - {match['title']} (Match Score: {match['match_score']:.1f}%)"
                    ):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Practice Areas:**")
                            st.write(match['practice_areas'])
                            
                            st.write("**Industry Experience:**")
                            st.write(match['industry_experience'])
                            
                        with col2:
                            st.write("**Languages:**")
                            st.write(match['languages'])
                            
                            st.write("**Location:**")
                            st.write(match['location'])
                            
                        st.write("**Relevant Skills:**")
                        for skill in match['relevant_skills'][:5]:  # Show top 5 skills
                            st.write(f"- {skill}")
                            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        💡 **About Rolodex AI:**  
        Powered by advanced AI technology, Rolodex AI matches you with Caravel Law attorneys based on their verified skills and expertise.
        """
    )

if __name__ == "__main__":
    main()
