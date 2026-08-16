import json
import re
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from .config import Config
from .models import ExamPlan, ExamQuestion
from .retriever import DocumentRetriever

class CognitivePlanner:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.2
        )
        self.retriever = DocumentRetriever()

        # Semantic queries designed to pull deep technical paragraphs per Bloom's level
        self.blooms_strategies = [
            {
                "level": "Recall",
                "query": "formal definitions, core terminology, dataset specifications, architectural notation",
                "instruction": "Ask the student to recall and define a specific technical mechanism, parameter, or term verbatim from the text."
            },
            {
                "level": "Understand",
                "query": "methodology workflow, system design, theoretical foundations, underlying principles",
                "instruction": "Ask the student to explain the core working principle or why a specific method functions in their own words."
            },
            {
                "level": "Apply",
                "query": "implementation details, hyperparameters, execution pipeline, practical constraints",
                "instruction": "Present a hypothetical scenario or use-case and ask the student how this mechanism/formula applies."
            },
            {
                "level": "Analyze",
                "query": "ablation studies, baseline comparisons, trade-offs, experimental error analysis",
                "instruction": "Ask the student to deconstruct, compare performance trade-offs, or analyze specific experimental/ablation findings."
            },
            {
                "level": "Evaluate",
                "query": "limitations, failure modes, assumptions, boundary conditions, future improvements",
                "instruction": "Ask the student to critically evaluate the validity of a claim, model limitation, or constraint discussed in the paper."
            }
        ]

    def _clean_json_string(self, raw_string: str) -> str:
        clean = re.sub(r"```json\s*", "", raw_string)
        clean = re.sub(r"```\s*$", "", clean)
        clean = clean.strip()
        try:
            json.loads(clean)
            return clean
        except json.JSONDecodeError:
            return re.sub(r'\\(?![/u"\\bfnrt])', r'\\\\', clean)

    def _extract_paper_topic(self, raw_text: str) -> str:
        """Fast, low-token extraction of the main paper title/topic."""
        snippet = raw_text[:2000]
        prompt = f"Identify the specific scientific paper title or core topic from this snippet. Output ONLY the short title/topic name:\n\n{snippet}"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().replace('"', '')

    def generate_exam_plan(self, raw_text: str) -> ExamPlan:
        # 1. Index document chunks
        self.retriever.build_index(raw_text)
        topic = self._extract_paper_topic(raw_text)
        print(f"Extracted Topic: {topic}")

        questions: List[ExamQuestion] = []

        # 2. Iterate through each Bloom's Level with targeted retrieval
        for strategy in self.blooms_strategies:
            level = strategy["level"]
            query = f"{topic}: {strategy['query']}"
            
            # Retrieve verified parent chunk
            context_chunk = self.retriever.retrieve_parent_context(query)
            if not context_chunk:
                continue

            print(f"Generating question for Bloom's level: [{level}]...")
            question_obj = self._generate_single_question(
                level=level,
                instruction=strategy["instruction"],
                context_snippet=context_chunk
            )
            questions.append(question_obj)

        return ExamPlan(topic=topic, questions=questions)

    def _generate_single_question(self, level: str, instruction: str, context_snippet: str) -> ExamQuestion:
        system_prompt = f"""
        You are a strict Academic Oral Examiner.
        You must formulate ONE examination question targeting Bloom's Taxonomy level: **{level}**.

        ### INSTRUCTION:
        {instruction}

        ### STRICT REQUIREMENTS:
        1. Base your question, rubric, and exemplar answer **ONLY** on the provided context snippet.
        2. Do not leak the answer inside the question.
        3. "grading_criteria" must be 1-2 complete sentences explaining what constitutes a correct answer.
        4. "exemplar_answer" must be a concise, ideal student response.
        5. "concepts" must list 2-4 key technical terms present in the snippet that must be included.

        Output ONLY valid JSON matching this schema:
        {{
          "type": "{level}",
          "question": "Question text...",
          "grading_rubric": {{
              "grading_criteria": "Sentence detailing required reasoning...",
              "concepts": ["keyword1", "keyword2"],
              "exemplar_answer": "Concise model response..."
          }}
        }}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context Snippet:\n{context_snippet}")
        ]

        response = self.llm.invoke(messages)
        clean_json = self._clean_json_string(response.content)
        data = json.loads(clean_json)

        # Inject the verified chunk as the context_snippet
        data["context_snippet"] = context_snippet

        return ExamQuestion(**data)