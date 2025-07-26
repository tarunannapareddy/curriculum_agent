import os
import json
import requests
from typing import Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.settings import Settings
from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel

# Brightdata API configuration
BRIGHTDATA_PROXY = "brd.superproxy.io:33335"
BRIGHTDATA_USER = "brd-customer-hl_b08cb01d-zone-real_time_search:ja7epdjc7a5t"
bright_data = False
# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@dataclass
class CurriculumInput:
    target_language: str  # e.g., "French", "Spanish"
    scenario: str  # e.g., "Cafe Order", "Hotel Check-in"

class CurriculumOutput(BaseModel):
    scenario_scene: str
    curriculum_questions: List[Dict[str, str]]
    correction_examples: List[Dict[str, str]]

class LlamaCurriculumAgent:
    def __init__(self):
        self.llm = GoogleGenAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        # Use Settings instead of ServiceContext for newer LlamaIndex versions
        Settings.llm = self.llm
        # Use local embeddings to avoid OpenAI API key requirement
        Settings.embed_model = "local:sentence-transformers/all-MiniLM-L6-v2"
        self.index = self._build_knowledge_index()
        
        # Pre-compute knowledge contexts for all language-scenario combinations
        print("🔄 Pre-computing knowledge contexts...")
        self.knowledge_cache = {}
        self.real_world_cache = {}
        self._precompute_contexts()
        print("✅ Knowledge contexts pre-computed successfully!")
        

        
    def _build_knowledge_index(self) -> VectorStoreIndex:
        """Build LlamaIndex with comprehensive knowledge base for language learning."""
        documents = []
        
        # Comprehensive knowledge base for different scenarios
        knowledge_base = {
            "cafe_order": {
                "French": {
                    "vocabulary": "bonjour, s'il vous plaît, merci, café, thé, eau, pain, croissant, combien, euros, commander, boire, manger, dessert, plat principal, entrée, addition, pourboire, spécialité, recommandation, allergie, végétarien, sans gluten, frais, bio, local, saisonnier, terroir, gastronomie",
                    "grammar": "Present tense, polite requests with 'je voudrais', questions with 'est-ce que', numbers, articles (le, la, les), negation with 'ne...pas', conditional tense for polite requests, relative pronouns, complex questions",
                    "interactions": "Greeting with 'bonjour', ordering with 'je voudrais', asking prices with 'combien', thanking with 'merci', detailed ordering, asking about ingredients, requesting modifications, asking for recommendations, discussing dietary restrictions, cultural conversation, expressing preferences",
                    "cultural_notes": "French cafes are social spaces. Say 'bonjour' when entering. Tipping is appreciated but not mandatory. French dining is leisurely. Lunch is typically 12-2 PM. Many cafes have outdoor seating. French cuisine emphasizes fresh, local ingredients. Wine is often served with meals."
                },
                "Spanish": {
                    "vocabulary": "hola, por favor, gracias, café, té, agua, pan, churros, cuánto, euros, uno, dos, tres, pedir, beber, comer, postre, plato principal, entrada, cuenta, propina, especialidad",
                    "grammar": "Present tense with regular verbs, simple questions, numbers 1-20, articles (el, la, los, las), negation, polite requests with 'podría'",
                    "interactions": "Greeting with 'hola', ordering with 'quisiera', asking prices with 'cuánto', thanking with 'gracias', detailed ordering, asking about ingredients, requesting modifications",
                    "cultural_notes": "Spanish cafes serve tapas. Lunch is typically served from 2-4 PM. Coffee is often served with a small glass of water. Spanish dining is social and leisurely. Tapas are meant to be shared."
                }
            },
            "hotel_checkin": {
                "French": {
                    "vocabulary": "hôtel, chambre, réservation, nom, passeport, clé, étage, ascenseur, wifi, téléphone, réception, réceptionniste, confirmer, annuler, tarif, petit-déjeuner, service, chambre simple/double, équipement, climatisation, vue, calme, bruit, problème technique, maintenance, concierge",
                    "grammar": "Present tense of 'avoir' and 'être', basic questions, numbers, possessive adjectives, past tense (passé composé), polite requests, time expressions, future tense for plans, subjunctive mood, complex conditional sentences, formal register, passive voice",
                    "interactions": "Basic check-in process, providing personal information, asking for room key, simple requests, confirming reservations, asking about hotel services, requesting changes, discussing preferences, complaining professionally, requesting changes, discussing preferences, problem resolution",
                    "cultural_notes": "Check-in is usually after 3 PM. Many hotels require passport for registration. French hotels often include breakfast. Room service is common in higher-end hotels. French hotels emphasize service quality. Concierge services are common in luxury hotels."
                },
                "Spanish": {
                    "vocabulary": "hotel, habitación, reserva, nombre, pasaporte, llave, piso, ascensor, wifi",
                    "grammar": "Present tense of 'tener' and 'ser', basic questions, numbers",
                    "interactions": "Basic check-in, providing information, asking for room key",
                    "cultural_notes": "Check-in typically starts at 2 PM. Spanish hotels often have siesta hours."
                }
            },
            "shopping": {
                "French": {
                    "vocabulary": "magasin, boutique, prix, cher, bon marché, taille, couleur, essayer, payer, carte, réduction, solde, marque, qualité, matière, style, mode, tendance, collection, artisanat, fait main, éthique, durable, vintage, exclusif, sur mesure, créateur",
                    "grammar": "Present tense, simple questions, numbers, basic adjectives, comparatives, superlatives, past tense, polite requests, complex conditionals, subjunctive, relative clauses, formal expressions",
                    "interactions": "Asking for items, checking prices, trying on clothes, basic payment, asking for discounts, discussing quality, comparing items, negotiating, discussing craftsmanship, ethical shopping, custom orders, cultural appreciation",
                    "cultural_notes": "French shopping is often in small boutiques. Sales happen in January and July. French fashion is world-renowned. Many shops close for lunch (12-2 PM). France values craftsmanship and quality over mass production."
                }
            }
        }
        
        # Convert knowledge base to documents for LlamaIndex
        for scenario, languages in knowledge_base.items():
            for language, content in languages.items():
                # Create detailed document for each scenario-language combination
                doc_text = f"""
                Scenario: {scenario.replace('_', ' ').title()}
                Target Language: {language}
                
                Vocabulary Focus: {content['vocabulary']}
                Grammar Focus: {content['grammar']}
                Interaction Patterns: {content['interactions']}
                Cultural Context: {content['cultural_notes']}
                
                Teaching Guidelines:
                - Use appropriate vocabulary for {language}
                - Focus on {content['grammar']} structures
                - Emphasize {content['interactions']} in {scenario.replace('_', ' ')} context
                - Include cultural notes: {content['cultural_notes']}
                """
                
                documents.append(Document(text=doc_text))
        

        
        return VectorStoreIndex.from_documents(documents)
    
    def get_available_scenarios_and_languages(self) -> tuple[list[str], list[str]]:
        """Get available scenarios and languages from the knowledge base"""
        scenarios = set()
        languages = set()
        
        # Extract from the knowledge base structure
        for doc_id, doc in self.index.docstore.docs.items():
            # Parse the document text to extract scenario and language
            text = doc.text
            if "Scenario:" in text:
                # Extract scenario name
                scenario_line = [line for line in text.split('\n') if line.strip().startswith('Scenario:')]
                if scenario_line:
                    scenario = scenario_line[0].replace('Scenario:', '').strip()
                    scenarios.add(scenario)
            
            if "Target Language:" in text:
                # Extract language name
                language_line = [line for line in text.split('\n') if line.strip().startswith('Target Language:')]
                if language_line:
                    language = language_line[0].replace('Target Language:', '').strip()
                    languages.add(language)
        
        # Convert to sorted lists
        scenarios_list = sorted(list(scenarios))
        languages_list = sorted(list(languages))
        
        return scenarios_list, languages_list
    
    def _precompute_contexts(self):
        """Pre-compute knowledge and real-world contexts for all language-scenario combinations"""
        # Get available scenarios and languages
        scenarios, languages = self.get_available_scenarios_and_languages()
        
        # Pre-compute knowledge contexts
        for scenario in scenarios:
            for language in languages:
                # Create cache key
                cache_key = f"{scenario}_{language}"
                
                # Pre-compute knowledge context
                try:
                    knowledge_context = self._query_knowledge_base(scenario, language)
                    self.knowledge_cache[cache_key] = knowledge_context
                except Exception as e:
                    print(f"Warning: Could not pre-compute knowledge for {cache_key}: {e}")
                    self.knowledge_cache[cache_key] = ""
                
                # Pre-compute real-world context
                try:
                    real_world_context = self._get_real_world_data(scenario, language)
                    self.real_world_cache[cache_key] = real_world_context
                except Exception as e:
                    print(f"Warning: Could not pre-compute real-world data for {cache_key}: {e}")
                    self.real_world_cache[cache_key] = ""
        
        print(f"📊 Pre-computed contexts for {len(scenarios)} scenarios × {len(languages)} languages = {len(self.knowledge_cache)} combinations")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring"""
        return {
            "knowledge_cache_size": len(self.knowledge_cache),
            "real_world_cache_size": len(self.real_world_cache),
            "cached_combinations": list(self.knowledge_cache.keys()),
            "cache_hit_rate": "100%" if len(self.knowledge_cache) > 0 else "0%"
        }
    
    def _fetch_brightdata_api(self, scenario: str, target_language: str) -> str:
        """Fetch real-world data from Brightdata API for the given scenario and language."""
        try:
            # Create query from language + scenario
            query = f"{target_language} {scenario}"
            
            # Construct the API URL (equivalent to curl command)
            url = f"https://www.google.com/search?q={query}&start=0&num=10"
            
            # Set up proxy configuration (equivalent to --proxy and --proxy-user)
            proxies = {
                "http": f"http://brd-customer-hl_b08cb01d-zone-real_time_search:ja7epdjc7a5t@brd.superproxy.io:33335",
                "https": f"http://brd-customer-hl_b08cb01d-zone-real_time_search:ja7epdjc7a5t@brd.superproxy.io:33335"
            }
            
            # Make the API request (equivalent to curl -v --compressed --proxy ... -k)
            response = requests.get(
                url,
                proxies=proxies,
                verify=False,  # -k flag equivalent
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept-Encoding": "gzip, deflate, br"  # --compressed equivalent
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.text
            else:
                print(f"Brightdata API returned status code: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"Error fetching data from Brightdata API: {e}")
            return ""
    
    def _parse_brightdata_response(self, api_response: str, scenario: str, target_language: str) -> Dict[str, str]:
        """Parse the Brightdata API response and extract relevant information."""
        try:
            # Use Gemini to parse and structure the API response
            prompt = f"""
            Parse the following Google Maps search results for {scenario} in {target_language} context.
            Extract relevant information and format it as JSON with the following structure:
            {{
                "menu_items": "Extracted menu items, prices, and services",
                "common_phrases": "Common phrases used in this context",
                "cultural_notes": "Cultural observations and local customs"
            }}
            
            API Response:
            {api_response[:2000]}  # Limit to first 2000 chars to avoid token limits
            
            Focus on authentic, real-world information that would be useful for language learners.
            """
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            parsed_data = json.loads(response_text)
            return parsed_data
            
        except Exception as e:
            print(f"Error parsing Brightdata response: {e}")
            return {
                "menu_items": "N/A",
                "common_phrases": "N/A", 
                "cultural_notes": "N/A"
            }
    
    def _get_real_world_data(self, scenario: str, target_language: str) -> str:
        """Get real-world data for authenticity."""
        
        if bright_data:
            api_response = self._fetch_brightdata_api(scenario, target_language)
            
            if api_response:
                parsed_data = self._parse_brightdata_response(api_response, scenario, target_language)
                
                return f"""
                Real-world context for {scenario} in {target_language}:
                Menu/Terms: {parsed_data.get('menu_items', 'N/A')}
                Common Phrases: {parsed_data.get('common_phrases', 'N/A')}
                Cultural Notes: {parsed_data.get('cultural_notes', 'N/A')}
                """
        
        # Use mock data
        real_world_data = {
            "cafe_order": {
                "French": {
                    "menu_items": "Café au lait (€3.50), Croissant (€1.20), Pain au chocolat (€1.30), Tarte Tatin (€4.50), Macaron (€2.00)",
                    "common_phrases": "Un café, s'il vous plaît. / L'addition, s'il vous plaît. / C'est combien? / Avez-vous du lait?",
                    "cultural_notes": "French cafes often have outdoor seating. Tipping is appreciated but not mandatory. Coffee is typically served in small cups."
                },
                "Spanish": {
                    "menu_items": "Café con leche (€2.80), Churros (€3.50), Tortilla española (€8.00), Paella (€15.00), Tapas (€3-8)",
                    "common_phrases": "Un café, por favor. / La cuenta, por favor. / ¿Cuánto cuesta? / ¿Tienen leche?",
                    "cultural_notes": "Spanish cafes serve tapas. Lunch is typically served from 2-4 PM. Coffee is often served with a small glass of water."
                }
            },
            "hotel_checkin": {
                "French": {
                    "hotel_terms": "Chambre simple (€80), Chambre double (€120), Suite (€200), Petit-déjeuner inclus, Vue sur la ville",
                    "common_phrases": "J'ai une réservation. / Pouvez-vous confirmer ma chambre? / À quel étage? / L'ascenseur, s'il vous plaît.",
                    "cultural_notes": "Check-in is usually after 3 PM. Many hotels require passport for registration. French hotels emphasize service quality."
                },
                "Spanish": {
                    "hotel_terms": "Habitación individual (€70), Habitación doble (€110), Suite (€180), Desayuno incluido, Vista a la ciudad",
                    "common_phrases": "Tengo una reserva. / ¿Puede confirmar mi habitación? / ¿En qué piso? / El ascensor, por favor.",
                    "cultural_notes": "Check-in typically starts at 2 PM. Spanish hotels often have siesta hours. Service is warm and personal."
                }
            }
        }
        
        scenario_key = scenario.lower().replace(" ", "_")
        language_data = real_world_data.get(scenario_key, {}).get(target_language, {})
        
        return f"""
        Real-world context for {scenario} in {target_language}:
        Menu/Terms: {language_data.get('menu_items', 'N/A')}
        Common Phrases: {language_data.get('common_phrases', 'N/A')}
        Cultural Notes: {language_data.get('cultural_notes', 'N/A')}
        """
    
    def _query_knowledge_base(self, scenario: str, target_language: str) -> str:
        """Query LlamaIndex for relevant knowledge using semantic search."""
        query = f"Provide vocabulary, grammar, and interaction guidelines for {scenario} scenario in {target_language}"
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
    
    def generate_curriculum(self, input_data: CurriculumInput) -> CurriculumOutput:
        """Generate curriculum using LlamaIndex for knowledge retrieval and Gemini for content generation."""
        
        # Get cached contextual knowledge
        cache_key = f"{input_data.scenario}_{input_data.target_language}"
        knowledge_context = self.knowledge_cache.get(cache_key, "")
        if not knowledge_context:
            # Fallback to real-time query if not in cache
            knowledge_context = self._query_knowledge_base(input_data.scenario, input_data.target_language)
        
        # Get cached real-world data
        real_world_context = self.real_world_cache.get(cache_key, "")
        if not real_world_context:
            # Fallback to real-time query if not in cache
            real_world_context = self._get_real_world_data(input_data.scenario, input_data.target_language)
        
        # Create comprehensive prompt for Gemini
        prompt = f"""
        Generate a language learning curriculum for {input_data.target_language}.
        
        Scenario: {input_data.scenario}
        
        Contextual Knowledge from LlamaIndex:
        {knowledge_context}
        
        Real-world Context:
        {real_world_context}
        
        Please generate a JSON response with the following structure:
        {{
            "scenario_scene": "A detailed description of the scenario setting and context",
            "curriculum_questions": [
                {{
                    "question": "A question to guide the AI in the conversation",
                    "expected_response": "What the AI should respond with"
                }}
            ],
            "correction_examples": [
                {{
                    "incorrect_phrase": "Common mistake a learner might make",
                    "correct_phrase": "The correct way to say it",
                    "explanation": "Gentle explanation of the correction"
                }}
            ]
        }}
        
        Make the content authentic to {input_data.target_language} culture and appropriate for language learners.
        Use the knowledge from LlamaIndex to ensure accuracy and cultural appropriateness.
        """
        
        try:
            # Generate content using Gemini
            model = genai.GenerativeModel('gemini-2.0-flash') # Using 2.0-flash for stability
            response = model.generate_content(prompt)
            
            # Extract JSON from markdown code block if present
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove ```
            response_text = response_text.strip()
            
            # Parse the JSON response
            curriculum_data = json.loads(response_text)
            return CurriculumOutput(**curriculum_data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error generating curriculum: {e}. Falling back to default.")
            return self._create_fallback_curriculum(input_data)
    
    def _create_fallback_curriculum(self, input_data: CurriculumInput) -> CurriculumOutput:
        """Create a fallback curriculum if generation fails."""
        
        # Create scenario scene based on inputs
        if input_data.scenario.lower() == "cafe order":
            scenario_scene = f"You are in a charming {input_data.target_language} café. The waiter approaches your table with a warm smile. Practice your {input_data.target_language} skills."
        elif input_data.scenario.lower() == "hotel check-in":
            scenario_scene = f"You are at the reception desk of a {input_data.target_language} hotel. The receptionist greets you. Practice your {input_data.target_language} skills."
        else:
            scenario_scene = f"You are in a {input_data.scenario.lower()} setting. Practice your {input_data.target_language} skills."
        
        # Create basic questions based on language and level
        if input_data.target_language == "French":
            questions = [
                {
                    "question": "How would you greet someone in this scenario?",
                    "expected_response": "Bonjour, monsieur/madame"
                },
                {
                    "question": "What would you like to order or request?",
                    "expected_response": "Je voudrais un café, s'il vous plaît" if "cafe" in input_data.scenario.lower() else "J'ai une réservation"
                }
            ]
            corrections = [
                {
                    "incorrect_phrase": "I want coffee",
                    "correct_phrase": "Je voudrais un café",
                    "explanation": "Use polite forms and proper articles in French"
                }
            ]
        elif input_data.target_language == "Spanish":
            questions = [
                {
                    "question": "How would you greet someone in this scenario?",
                    "expected_response": "Hola, señor/señora"
                },
                {
                    "question": "What would you like to order or request?",
                    "expected_response": "Quisiera un café, por favor" if "cafe" in input_data.scenario.lower() else "Tengo una reserva"
                }
            ]
            corrections = [
                {
                    "incorrect_phrase": "I want coffee",
                    "correct_phrase": "Quisiera un café",
                    "explanation": "Use polite forms and proper articles in Spanish"
                }
            ]
        else:
            questions = [
                {
                    "question": "How would you greet someone in this scenario?",
                    "expected_response": "Use appropriate greetings for the context and language level."
                },
                {
                    "question": "What would you like to order or request?",
                    "expected_response": "Make a simple request using basic vocabulary and grammar."
                }
            ]
            corrections = [
                {
                    "incorrect_phrase": "I want coffee",
                    "correct_phrase": "Use polite forms in the target language",
                    "explanation": "Use polite forms and proper articles in the target language."
                }
            ]
        
        return CurriculumOutput(
            scenario_scene=scenario_scene,
            curriculum_questions=questions,
            correction_examples=corrections
        )