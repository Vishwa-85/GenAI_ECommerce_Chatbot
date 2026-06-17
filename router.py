from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
import numpy as np


encoder = HuggingFaceEncoder(
    name="sentence-transformers/all-MiniLM-L6-v2"
)

faq = Route(
    name='faq',
    utterances=[
        "What is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "How long does it take to process a refund?",
        "Are there any ongoing sales or promotions?",
        "Can I cancel or modify my order after placing it?"
    ]
)

sql = Route(
    name='sql',
    utterances=[
        "I want to buy Nike shoes with 50% discount",
        "Show me the shoes under Rs 3000",
        "Are there any Puma shoes on sale?",
        "What is the price of Adidas running shoes?",
        "Do you have any formal shoes with size 10?"
    ]
)

router = SemanticRouter(
    routes=[faq, sql],
    encoder = encoder,
    auto_sync="local",
    top_k=1
)

if __name__ == "__main__":
    print(router("Show me the shoes under Rs 5000?").name)

  