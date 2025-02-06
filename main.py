import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

def clean_skills_data(skills_file: str) -> pd.DataFrame:
    """
    Clean the skills matrix data by removing test entries and specific individuals.
    
    Args:
        skills_file (str): Path to the skills matrix CSV file
    
    Returns:
        pd.DataFrame: Cleaned skills matrix
    """
    # Read the skills matrix
    df = pd.read_csv(skills_file)
    
    # Remove entries containing 'test' in any field (case insensitive)
    df = df[~df.apply(lambda x: x.astype(str).str.contains('test', case=False, na=False)).any(axis=1)]
    
    # Remove entries for specific individuals
    names_to_remove = ['Ankita Avadhani', 'Tania']
    df = df[~df['Submitter Name'].str.contains('|'.join(names_to_remove), case=False, na=False)]
    
    return df

class LawyerMatchingSystem:
    def __init__(self):
        """Initialize the LawyerMatchingSystem with necessary components"""
        self.skills_data = None
        self.lawyer_data = None
        self.skill_vectors = None
        self.lawyer_map = {}  # Map between skills responses and lawyer profiles
        
    def load_data(self) -> None:
        """Load and preprocess the skills and lawyer data"""
        # Load and clean skills data
        self.skills_data = clean_skills_data('skills_responses.csv')
        self.lawyer_data = pd.read_csv('BD_Caravel.csv')
        
        # Create mapping between skills responses and lawyer profiles using name
        self.lawyer_map = {
            name: idx for idx, name in enumerate(self.skills_data['Submitter Name'])
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
        
    def analyze_legal_need(self, query: str) -> Dict[str, Any]:
        """Mock analysis of legal needs based on keywords"""
        # Simple keyword-based analysis for now
        query_lower = query.lower()
        
        # Default skills based on common keywords
        skills_map = {
            'corporate': ['Commercial Contracts', 'Corporate Bylaws', 'M&A'],
            'contract': ['Commercial Contracts', 'Master Services Agreements', 'Professional Services Agreements'],
            'intellectual property': ['Intellectual Property Protection', 'Patent Portfolio Management', 'Trademark Law'],
            'employment': ['Employment Agreements', 'Labour and Union', 'Employment-based Immigration'],
            'privacy': ['Privacy Compliance', 'Data Protection', 'Cross-Border Privacy Compliance']
        }
        
        # Find matching skills based on keywords
        matched_skills = []
        for key, skills in skills_map.items():
            if key in query_lower:
                matched_skills.extend(skills)
        
        # If no specific matches, use general business skills
        if not matched_skills:
            matched_skills = ['Commercial Contracts', 'Corporate Bylaws', 'Professional Services Agreements']
        
        return {
            'primary_skills': list(set(matched_skills)),
            'required_experience': 5,
            'industry_focus': ['Technology', 'General Business'],
            'practice_areas': ['Corporate Commercial', 'Business Law']
        }
    
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
            # Try to find email in profile data (handle different possible column names)
            email_columns = ['Email', 'email', 'E-mail', 'e-mail']
            lawyer_matches = pd.DataFrame()
            
            for email_col in email_columns:
                if email_col in self.lawyer_data.columns:
                    lawyer_matches = self.lawyer_data[
                        self.lawyer_data[email_col].str.lower() == lawyer_email.lower()
                    ]
                    if not lawyer_matches.empty:
                        break
            
            # If no email match, try matching by name
            if lawyer_matches.empty:
                submitter_name = self.skills_data.iloc[idx]['Submitter Name']
                if 'First Name' in self.lawyer_data.columns and 'Last Name' in self.lawyer_data.columns:
                    name_parts = submitter_name.split()
                    if len(name_parts) >= 2:
                        lawyer_matches = self.lawyer_data[
                            (self.lawyer_data['First Name'].str.lower() == name_parts[0].lower()) &
                            (self.lawyer_data['Last Name'].str.lower() == name_parts[-1].lower())
                        ]
            
            if len(lawyer_matches) == 0:
                continue
                
            lawyer_profile = lawyer_matches.iloc[0]
            
            # Calculate match percentage
            match_score = similarities[idx] * 100
            
            # Create match info
            match_info = {
                'name': f"{lawyer_profile['First Name']} {lawyer_profile['Last Name']}",
                'email': lawyer_profile.get('email', lawyer_profile.get('Email', '')),  # Try both capitalizations
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
                requirements = matcher.analyze_legal_need(legal_need)
                
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
                st.subheader("👥 Matching Attorneys")
                
                # Add match score explanation
                st.markdown("""
                    The match score indicates how well an attorney's expertise aligns with your specific legal needs. 
                    It considers their self-assessed proficiency across various legal skills and practice areas.
                """)
                
                # Display matches without showing they're ranked
                for match in matches:
                    with st.expander(
                        f"{match['name']} - {match['title']}"
                    ):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Practice Areas:**")
                            st.write(match['practice_areas'])
                            
                            st.write("**Industry Experience:**")
                            st.write(match['industry_experience'])
                            
                        with col2:
                            st.write("**Email:**")
                            st.write(match['email'])
                                
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
