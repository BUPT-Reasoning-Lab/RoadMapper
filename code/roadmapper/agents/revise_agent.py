from openai import AsyncOpenAI
from .utils import ExtractMarkdownUtil


async def ReviseAgent( client: AsyncOpenAI, model: str, project: str, roadmap: str, split_language: str="English", logic_critique_result: str=None, granularity_critique_result: str=None) -> tuple:
    """Revise roadmap based on critique results"""
    system_prompt = f"""
You are an experienced research expert, specialized in revising research problem roadmaps based on improvement suggestions.
For a research problem, a proposed roadmap (in Markdown format), and improvement suggestions provided by the user (in Markdown format), you need to revise the roadmap based on the improvement suggestions provided by the user.

Please strictly follow the requirements below:
1. Improvement suggestions include unreasonable logical dependencies and unreasonable granularity, with specific formats and meanings as follows:
   - If the parent-child dependency is unreasonable, the improvement suggestion format is: [x.x.x xxx -> x.x.x xxx] [Parent-child dependency is unreasonable], because xxx, you should revise the roadmap by xxx.
   - If the sibling dependency is unreasonable, the improvement suggestion format is: [x.x.x xxx -> x.x.x xxx] [Sibling dependency is unreasonable], because xxx, you should revise the roadmap by xxx.
   - If a node has no children but its content is too much to need to be refined, the improvement suggestion format is: [x.x.x xxx] [Too Brief], because xxx, you should revise the roadmap by xxx.
   - If a node has children but is divided too finely, the improvement suggestion format is: [x.x.x xxx] [Too Detailed], because xxx, you should revise the roadmap by xxx.
2. For each improvement suggestion provided by the user, you need to determine whether the suggestion is helpful for improving the roadmap. Adopt suggestions that are helpful and revise the roadmap accordingly; reject unreasonable or meaningless suggestions.
3. For each improvement suggestion, output a comment (regardless of whether it is adopted), and present all comments in a Markdown unordered list.
4. If an improvement suggestion is adopted, the output format should be: [Accepted][xxx] I have revised the roadmap by xxx.
5. If an improvement suggestion is rejected, the output format should be: [Rejected][xxx] I have rejected the suggestion because xxx.
6. Roadmap format requirements, make sure to strictly follow:
   - Use Markdown format for output, with each line as a node (including level, index, and title, separated by spaces).
   - Use different heading levels (#, ##, ###, etc.) to indicate the node level and the hierarchical structure of the roadmap.
   - Use indices like 1.1.1 to indicate the node's position in the roadmap.
   - The title should use the format [xxx], where xxx is the node content.
7. Use two sets of ```markdown``` to wrap the output results: the first wraps the revised roadmap, and the second wraps the comments.
8. The roadmap content should be answered in {split_language}.

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

```markdown
- [Accepted][xxx] I have revised the roadmap by xxx.
- [Rejected][xxx] I have rejected the suggestion because xxx.
...
```
    """
    user_prompt = f"""
The research problem provided by the user is: {project}
The roadmap provided by the user is: {roadmap}
The logic dependency check result provided by the user is: {logic_critique_result}
The granularity check result provided by the user is: {granularity_critique_result}
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    matches = ExtractMarkdownUtil(response.choices[0].message.content, "markdown")
    if matches and len(matches) == 2:
        return matches[0], matches[1], response.model_dump()
    else:
        raise Exception(f"ReviseAgent failed, returned empty.\n{response.choices[0].message.content}")

