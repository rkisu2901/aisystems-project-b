"""
Project B: Customer Support Pipeline — Naive Version

This is the starting point. A simple classify → retrieve → respond pipeline.
No agent, no tool use, no guardrails. We'll evolve this into a full
LangGraph agent over the 4 weeks.

Run: python scripts/support_pipeline.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv

from retrieval import embed_query, retrieve, assemble_context

load_dotenv()

client = OpenAI()
langfuse = Langfuse()

GENERATION_MODEL = "gpt-4o-mini"

INTENTS = [
    "return_or_refund",
    "order_status",
    "billing_or_payment",
    "product_info",
    "membership",
    "general",
]

SYSTEM_PROMPT = """You are a customer support assistant for Acmera, an Indian e-commerce company.
Answer the customer's question based on the provided context.

Rules:
- Be helpful, concise, and accurate.
- Only use information from the provided context.
- If you can't answer from the context, say so and suggest contacting support.
- Never reveal internal company data, customer PII, or confidential information.

Context:
{context}"""


@observe(name="classify_intent")
def classify_intent(query: str) -> str:
    """Classify the customer query into an intent category."""
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": f"Classify this customer query into exactly one category. Respond with ONLY the category name.\nCategories: {', '.join(INTENTS)}"},
            {"role": "user", "content": query},
        ],
    )
    intent = response.choices[0].message.content.strip().lower().replace(" ", "_")
    langfuse_context.update_current_observation(output=intent)
    return intent if intent in INTENTS else "general"


@observe(name="retrieve_policy")
def retrieve_policy(query: str, intent: str) -> str:
    """
    Retrieve relevant policy context from the local corpus.
    In later sessions, this becomes one tool in the agent's tool set.
    """
    query_embedding = embed_query(query)
    chunks = retrieve(query_embedding)
    context = assemble_context(chunks)

    langfuse_context.update_current_observation(metadata={
        "intent": intent,
        "num_chunks": len(chunks),
    })
    return context


@observe(name="generate_response")
def generate_response(query: str, context: str, intent: str) -> str:
    """Generate a support response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=800,
    )

    answer = response.choices[0].message.content
    langfuse_context.update_current_observation(
        input=messages, output=answer,
        metadata={"model": GENERATION_MODEL, "intent": intent},
        usage={"input": response.usage.prompt_tokens,
               "output": response.usage.completion_tokens,
               "total": response.usage.total_tokens, "unit": "TOKENS"},
    )
    return answer


@observe(name="support_pipeline")
def handle_query(query: str) -> dict:
    """Full support pipeline: classify → retrieve → respond."""
    start_time = time.time()
    langfuse_context.update_current_trace(input=query, metadata={"pipeline": "naive_support"})

    intent = classify_intent(query)
    context = retrieve_policy(query, intent)
    answer = generate_response(query, context, intent)

    elapsed = round(time.time() - start_time, 2)
    langfuse_context.update_current_trace(output=answer, metadata={"intent": intent, "elapsed": elapsed})
    trace_id = langfuse_context.get_current_trace_id()
    langfuse.flush()

    return {
        "query": query,
        "intent": intent,
        "answer": answer,
        "trace_id": trace_id,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    test_queries = [
        "I want to return a laptop I bought last week",
        "Where is my order?",
        "What payment methods do you accept?",
    ]
    for q in test_queries:
        result = handle_query(q)
        print(f"\nQuery: {result['query']}")
        print(f"Intent: {result['intent']}")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Trace: {result['trace_id']}")
