from openai import AsyncOpenAI
from .utils import ExtractMarkdownUtil


async def InitAgent(client: AsyncOpenAI, model: str, project: str, split_language: str="English") -> tuple:
    """Generate initial roadmap by large model without knowledge base"""
    system_prompt = f"""
You are an experienced research expert, specialized in designing research problem roadmaps.
For a given research problem provided by the user, you need to design a roadmap to guide the user in solving the problem step by step.

Please strictly follow the requirements below:
1. Output a structured, multi-level roadmap that details and progressively explains the steps and sub-steps to solve the research problem.
2. Roadmap format requirements, make sure to strictly follow:
   - Use Markdown format for output, with each line as a node (including level, index, and title, separated by spaces).
   - Use different heading levels (#, ##, ###, etc.) to indicate the node level and the hierarchical structure of the roadmap.
   - Use indices like 1.1.1 to indicate the node's position in the roadmap.
   - The title should use the format [xxx], where xxx is the node content.
3. Enclose the output in ```markdown```.
4. The roadmap content should be answered in {split_language}.

Example output format:
```markdown
# 1 [Main Step]
## 1.1 [Sub-step]
## 1.2 [Sub-step]
# 2 [Main Step]
## 2.1 [Sub-step]
### 2.1.1 [Sub-step]
### 2.1.2 [Sub-step]
...
```
    """
    
    user_prompt = f"""
The research problem provided by the user is: {project}
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    matches = ExtractMarkdownUtil(response.choices[0].message.content, "markdown")
    initial_roadmap = matches[0] if matches else ""

    return initial_roadmap, response.model_dump()

