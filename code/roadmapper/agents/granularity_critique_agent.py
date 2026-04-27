from openai import AsyncOpenAI
from .utils import ExtractMarkdownUtil


async def GranularityCritiqueAgent(client: AsyncOpenAI, model: str, project: str, roadmap: str) -> tuple:
    """Check granularity appropriateness of roadmap nodes"""
    system_prompt = """
You are an experienced research expert, specialized in analyzing the granularity of research problem roadmaps.
For a research problem and a proposed roadmap (in Markdown format), you need to determine whether the granularity of each node in the roadmap is appropriate.

Please strictly follow the requirements below:
1. The nodes to be analyzed include: nodes with children (whether the child nodes can fully explain and implement the complete content of the node) and nodes without children (whether the node content is too much that needs to be refined).
2. Carefully analyze each node in the roadmap to determine if there are any unreasonable granularities.
3. Do not comment on reasonable granularities; for unreasonable ones, output a comment in Markdown unordered list format.
4. If a node has no children but its content is too much to need to be refined, the comment output format is: [x.x.x xxx] [Too Brief], because xxx, you should revise the roadmap by xxx.
5. If a node has children but is divided too finely, the comment output format is: [x.x.x xxx] [Too Detailed], because xxx, you should revise the roadmap by xxx.
6. Enclose the output in ```markdown```.

Example output format:
```markdown
- [1.1.1 xxx] [Too Brief], because xxx, you should revise the roadmap by xxx.
- [1.1.2 xxx] [Too Detailed], because xxx, you should revise the roadmap by xxx.
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
    granularity_critique_result = matches[0] if matches else ""
    
    return granularity_critique_result, response.model_dump()

