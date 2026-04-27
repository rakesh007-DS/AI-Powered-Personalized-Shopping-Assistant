from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from rag_system import ProductRAG

class ShoppingAssistantAgent:
    def __init__(self, fine_tuned_model_path: str, rag_system: ProductRAG, openai_api_key: str):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=openai_api_key)
        self.rag = rag_system
        self.user_profiles = {}
        self._setup_prompts()

    def _setup_prompts(self):
        self.recommendation_prompt = PromptTemplate(
            input_variables=["query", "profile", "products"],
            template="""You are an expert shopping assistant. User query: "{query}"

User profile: {profile}

Available products: {products}

Recommend TOP 3 products with:
1. Product name & price
2. Why it matches user needs
3. Key features
4. Call-to-action

Be friendly and conversational!"""
        )

    def get_personalized_recommendations(self, user_id: str, query: str) -> str:
        profile = self.user_profiles.get(user_id, {"preferences": "general"})
        products = self.rag.retrieve_products(query)

        chain = LLMChain(llm=self.llm, prompt=self.recommendation_prompt)
        response = chain.run(
            query=query,
            profile=str(profile),
            products=str(products[:5])
        )
        return response
