import chromadb
import json
from openai import AsyncOpenAI
from .utils import ExtractMarkdownUtil


async def CreateTechLib(client: AsyncOpenAI, model: str, project: str, chroma_client: chromadb.HttpClient, collection_name: str, project_embedding_file_path: str, n_results: int=10, split_language: str="English") -> tuple:
    """Generate skill point repository for improving roadmap"""
    collection = chroma_client.get_collection(name=collection_name)

    with open(project_embedding_file_path, "r") as f:
        project_embedding = json.load(f)

    retrieval_results = collection.query(
        query_embeddings=[project_embedding],
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    formatted_retrieval_results = []
    for i in range(len(retrieval_results.get("ids", [[]])[0])):
        skill_info = {
            "name": retrieval_results.get("metadatas", [[]])[0][i].get(
                "skill_point_name", ""
            ),
            "description": retrieval_results.get("metadatas", [[]])[0][i].get(
                "skill_point_description", ""
            ),
        }
        formatted_retrieval_results.append(skill_info)

    system_prompt = f"""
You are an experienced research expert, specialized in analyzing research problems and identifying key skills.
For a given research problem provided by the user, you need to analyze and output several key skills that are essential for solving the problem.

Please strictly follow the requirements below:
1. Analyze the research problem and output several key skills that are essential for solving the problem.
2. Output format requirements:
   - Use JSON list format for output, with each item being a skill point including name and description fields.
   - `name`: The name of the skill point (brief description).
   - `description`: The detailed explanation of the skill point (explains the role of the skill point in solving the problem).
   - Enclose the output in ```json```.
3. The skills should be answered in {split_language}.

Example output format:
```json
[
    {{"name": "xxx", "description": "xxx"}},
    {{"name": "xxx", "description": "xxx"}},
    ...
]
```
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"The research problem provided by the user is: {project}",
            },
        ],
    )

    matches = ExtractMarkdownUtil(response.choices[0].message.content, "json")
    ai_created_skills = json.loads(matches[0] if matches else "[]")

    combined_skills = formatted_retrieval_results + ai_created_skills
    
    return combined_skills, response.model_dump()


async def KnowledgeAgent(client: AsyncOpenAI, model: str, project: str, roadmap: str, techLib: list, split_language: str="English") -> tuple:
    """Improve roadmap based on skill point repository"""
    system_prompt = f"""
You are an experienced research expert, specialized in optimizing research problem roadmaps based on skill point repositories.
For a research problem, a current roadmap (in Markdown format), and a skill point repository provided by the user, you need to optimize the roadmap based on the research problem, the current roadmap, and the skill point repository.

Please strictly follow the requirements below:
1. Carefully analyze each skill point in the skill point repository (each skill point includes its name and description) according to the research problem and the current roadmap.
2. Determine which skill points are helpful for improving the roadmap, insert the names of helpful skill points as a step node into appropriate positions in the roadmap (names can be adapted to the problem as needed), and ignore unhelpful skill points.
3. Ensure the correct format of the roadmap during insertion, maintaining the logical order and continuity of node indices.
4. Roadmap format requirements, make sure to strictly follow:
   - Use Markdown format for output, with each line as a node (including level, index, and title, separated by spaces).
   - Use different heading levels (#, ##, ###, etc.) to indicate the node level and the hierarchical structure of the roadmap.
   - Use indices like 1.1.1 to indicate the node's position in the roadmap.
   - The title should use the format [xxx], where xxx is the node content.
5. Enclose the output in ```markdown```.
6. The roadmap content should be answered in {split_language}.

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
The roadmap provided by the user is: {roadmap}
The skill points provided by the user is: {json.dumps(techLib, ensure_ascii=False)}
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    matches = ExtractMarkdownUtil(response.choices[0].message.content, "markdown")
    knowledge_enhanced_roadmap = matches[0] if matches else ""
    
    return knowledge_enhanced_roadmap, response.model_dump()

