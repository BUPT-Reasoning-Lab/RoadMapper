from openai import AsyncOpenAI
from .utils import ExtractMarkdownUtil


async def LogicCritiqueAgent(client: AsyncOpenAI, model: str, project: str, roadmap: str) -> tuple:
    """Check logical dependencies between roadmap nodes"""
    system_prompt = """
You are an experienced research expert, specialized in analyzing the logical dependencies between nodes in research problem roadmaps.
For a research problem and a proposed roadmap (in Markdown format), you need to determine whether the logical dependencies between nodes in the roadmap are reasonable.

Please strictly follow the requirements below:
1. There are two types of dependencies: parent-child (child nodes are more detailed descriptions of the parent) and sibling (subsequent siblings are follow-up steps or extensions of previous siblings).
2. Carefully analyze every pair of parent-child and adjacent sibling nodes in the roadmap to identify any unreasonable dependencies.
3. Do not comment on reasonable dependencies; for unreasonable ones, output a comment in Markdown unordered list format.
4. If the parent-child dependency is unreasonable, the comment output format is: [x.x.x xxx -> x.x.x xxx] [Parent-child dependency is unreasonable], because xxx, you should revise the roadmap by xxx.
5. If the sibling dependency is unreasonable, the comment output format is: [x.x.x xxx -> x.x.x xxx] [Sibling dependency is unreasonable], because xxx, you should revise the roadmap by xxx.
6. Enclose the output in ```markdown```.

Example output format:
```markdown
- [1.1 xxx -> 1.1.2 xxx] [Parent-child dependency is unreasonable], because xxx, you should revise the roadmap by xxx.
- [1.1.1 xxx -> 1.1.2 xxx] [Sibling dependency is unreasonable], because xxx, you should revise the roadmap by xxx.
```
    """
    user_prompt = f"""
The research problem provided by the user is: {project}
The roadmap provided by the user is: {roadmap}
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    matches = ExtractMarkdownUtil(response.choices[0].message.content, "markdown")
    logic_critique_result = matches[0] if matches else ""
    
    return logic_critique_result, response.model_dump()

