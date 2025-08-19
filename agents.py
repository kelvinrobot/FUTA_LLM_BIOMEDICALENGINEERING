# bioinformatics_ai/agents.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from autogen import AssistantAgent, UserProxyAgent
from config import LLM_MODEL, CONFIDENCE_THRESHOLD, VECTORSTORE_DIR
from rag import RAGAgent
import os
import sys

#  Ensure Hugging Face cache is in a writable directory (important on HF Spaces)
if "HF_HOME" not in os.environ:
    hf_cache = "/home/user/.cache/huggingface"
    os.environ["HF_HOME"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache, "transformers")
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache, "hub")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# Load BioMistral once
class BioMistralModel:
    def __init__(self, model_name=LLM_MODEL, device=None):
        print(f"[BioMistralModel] Loading model: {model_name}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def generate_answer(self, query: str) -> str:
        prompt = f"You are a helpful bioinformatics tutor. Answer clearly:\n\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.split("Answer:", 1)[-1].strip()


# Formatting Agent
class FormattingAgent(AssistantAgent):
    def __init__(self, name="FormattingAgent", **kwargs):
        super().__init__(name=name, **kwargs)

    def format_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned


# Tutor Agent
class TutorAgent(AssistantAgent):
    def __init__(self, name="TutorAgent", **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = BioMistralModel()
        self.format_agent = FormattingAgent()
        self.rag_agent = RAGAgent(vectorstore_dir=str(VECTORSTORE_DIR))  # safe conversion

    def process_query(self, query: str) -> str:
        print(f"[TutorAgent] Received query: {query}")
        
        answer = self.model.generate_answer(query)
        confidence = self.estimate_confidence(answer)

        print(f"[TutorAgent] Confidence: {confidence:.2f}")
        if confidence < CONFIDENCE_THRESHOLD:
            print("[TutorAgent] Confidence low, but still using BioMistral (RAG unused).")

        return self.format_agent.format_text(answer)

    def estimate_confidence(self, answer: str) -> float:
        length = len(answer.strip())
        if length > 100:
            return 0.9
        elif length > 50:
            return 0.75
        else:
            return 0.5


# User Agent
class BioUser(UserProxyAgent):
    def __init__(self, name="BioUser", **kwargs):
        #  disable docker-based execution (not available in HF Spaces)
        kwargs.setdefault("code_execution_config", {"use_docker": False})
        super().__init__(name=name, **kwargs)
