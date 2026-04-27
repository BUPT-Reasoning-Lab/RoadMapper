import re
from openai import AsyncOpenAI


async def EvaluateAgent(evaluate_client: AsyncOpenAI, evaluate_model: str, project: str, roadmap: str) -> tuple:
    """Evaluate roadmap quality and return score (0-100), eval_reason, and model_response"""
    system_prompt = """You are an experienced research expert, specialized in evaluating the quality of research roadmaps.
For a research problem and a proposed roadmap (in Markdown format) used to solve the research problem, you need to evaluate the roadmap from multiple dimensions below and provide objective and fair scores and detailed analysis:
1. Logic Structure: Evaluate the logical coherence of the roadmap, including whether the dependencies between nodes are reasonable, whether the step order is logical, and whether there are conflicts or contradictions between different parts.
2. Granularity Degree: Evaluate the rationality of task decomposition granularity, including whether there are too macroscopic or too microscopic nodes, whether the sub-task division is balanced, etc.
3. Topic Relevance: Evaluate the relevance of the roadmap to the input research problem, including whether the roadmap fully covers the core content required to solve the problem, whether there are redundant nodes unrelated to the topic, and whether the key technical points are fully reflected, etc.
4. Completeness: Evaluate the richness and completeness of the roadmap content, including whether there are missing key nodes or steps, and whether all the content required to solve the problem is fully covered.

Based on the four dimensions above, provide an overall assessment using the following 100-point scale (percentage system, 0-100):
0-19: Very poor quality, cannot be used.
20-39: Poor quality, needs significant improvement and is not recommended as is.
40-59: Acceptable but still needs improvement before use.
60-79: Good quality and can be used with minor adjustments.
80-100: Excellent quality that are perfectly perfect in all four dimensions and can be directly used.

Output Format Requirements:
You must answer in English and enclose your score within <eval_score> tags and your detailed reasoning within <eval_reason> tags. For example:

<eval_score>68</eval_score>
<eval_reason>Your detailed analysis covering all four dimensions...</eval_reason>

Please strictly follow the requirements above and provide the output in the specified format."""

    user_prompt = f"""The research problem is:
{project}

The roadmap is:
{roadmap}"""

    response = await evaluate_client.chat.completions.create(
        model=evaluate_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    score_match = re.search(r"<eval_score>\s*(\d+)\s*</eval_score>", content, re.DOTALL)
    if not score_match:
        raise ValueError(f"Failed to parse model output [score]: {content}")
    
    score = int(score_match.group(1))
    
    # Extract eval_reason if present
    eval_reason_match = re.search(r"<eval_reason>\s*(.*?)\s*</eval_reason>", content, re.DOTALL)
    eval_reason = eval_reason_match.group(1).strip() if eval_reason_match else ""
    
    return score, eval_reason, response.model_dump()

