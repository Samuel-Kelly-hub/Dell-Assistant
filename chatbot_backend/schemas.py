from pydantic import BaseModel


class InformationGathererResponse(BaseModel):
    """Agent 1 assesses info sufficiency and classifies the question in one call.

    When has_enough_info is False: follow_up_question is populated,
    classified_question is an empty string.
    When has_enough_info is True: classified_question is populated,
    follow_up_question is an empty string.
    """
    has_enough_info: bool
    follow_up_question: str
    reasoning: str
    classified_question: str


class ContextSufficiencyAssessment(BaseModel):
    """Agent 3 checks whether the RAG results are sufficient."""
    is_sufficient: bool
    information_gap: str  # Description of what information is missing (empty if sufficient)
    reasoning: str


class FormulatedAnswer(BaseModel):
    """Agent 4 composes the final answer."""
    answer: str
    confidence: str
    sources_used: str


class UserFeedback(BaseModel):
    """Classifies the user's satisfaction response."""
    is_satisfied: bool
    is_uncertain: bool


class RAGRetrieverQuery(BaseModel):
    """Agent 2 formulates a search query for RAG retrieval."""
    search_query: str
    reasoning: str


class ClarificationAssessment(BaseModel):
    """Assesses whether user's clarification warrants another RAG attempt."""
    is_actionable: bool
    information_gap: str
    reasoning: str


class TOCAnalysis(BaseModel):
    """PDF fallback agent analyses first pages for a table of contents."""
    has_toc: bool
    relevant_pages: list[int]  # 1-indexed page numbers relevant to the question
    most_relevant_section_title: str  # Title of the most relevant section
    reasoning: str
