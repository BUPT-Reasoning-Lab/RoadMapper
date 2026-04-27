<div align="center">
    <h1>RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems</h1>
</div>


# 1 Introduction
The data and code for the paper `RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems`
## 1.1 Abstract
People commonly leverage structured content to accelerate knowledge acquisition and research problems solving. Among these, roadmaps guide researchers through hierarchical subtasks to solve complex research problems step by step. Despite progress in structured content generation, the **roadmap generation task** has remained unexplored. To bridge this gap, we introduce **RoadMap**, a novel benchmark designed to evaluate the ability of large language models (LLMs) to construct high-quality roadmaps for solving complex research problems. Based on this, we identify three limitations of LLMs: (1) lack of professional knowledge; (2) unreasonable task decomposition; (3) disordered logical relationships. To address these challenges, we propose **RoadMapper**, an LLM-based multi-agent system that decomposes the research roadmap generation task into three key stages (i.e., initial generation, knowledge augmentation, and iterative "critique-revise-evaluate"). Extensive experiments demonstrate that RoadMapper can improve LLMs' ability for roadmap generation, enhancing average performance by more than 8% while saving 71% of the time required by human experts, highlighting its effectiveness and application potential. Our data and code are available at https://anonymous.4open.science/r/RoadMapper.


## 1.2 Repository Structure
The repository is organized as follows:
```markdown
root/
├── chroma/                                     # ChromaDB related scripts and configuration
│   ├── chroma.py                               # ChromaDB operations (create, test, delete collections)
│   ├── config/
│   │   └── chroma.yaml                         # ChromaDB configuration file
│   └── start_chroma.sh                         # Script to start Chroma server
├── code/ 
│   ├── dpo/                                    # DPO (Direct Preference Optimization) related files
│   │   ├── qwen3-dpo.yaml                      # DPO training configuration
│   │   ├── qwen3-infer.sh                      # DPO inference script
│   │   └── qwen3-merge.yaml                    # DPO model merge configuration
│   ├── experiments/                            # Batch processing scripts for experiments
│   │   ├── infer.py                            # Batch inference pipeline for RoadMapper
│   │   └── eval.py                             # Batch evaluation pipeline for RoadMapper
│   ├── roadmapper/                             # Core RoadMapper multi-agent system
│   │   ├── agents/                             # Agent modules
│   │   │   ├── evaluate_agent.py               # Evaluate agent for roadmap quality assessment
│   │   │   ├── granularity_critique_agent.py   # Granularity Critique agent
│   │   │   ├── init_agent.py                   # Init agent
│   │   │   ├── knowledge_agent.py              # Knowledge agent
│   │   │   ├── logic_critique_agent.py         # Logic Critique agent
│   │   │   ├── revise_agent.py                 # Revise agent
│   │   │   └── utils.py                        # Utility functions for agents
│   │   ├── direct_prompting.py                 # Direct prompting baseline implementation
│   │   ├── force_n_round.py                    # Force N rounds iteration implementation
│   │   └── roadmapper_retry.py                 # Main RoadMapper pipeline with retry mechanism
│   └── scripts/                                # Utility scripts during markdown useful contents extraction when constructing RoadMap
│       ├── extract-content-before.py           # Content extraction utility 
│       └── extract-useful-only.py              # Useful content extraction utility
├── data/
│   ├── core-research-question-embeddings/      # Core research question embedding files (used by Knowledge Agent for skill point retrieval)
│   ├── golden-roadmap/                         # Gold standard roadmap files (used for evaluation)
│   └── skill-points-embeddings/                # Skill points embeddings files (used by Knowledge Agent for skill point retrieval)
├── dataset/
│   ├── dataset-split-cn.json                   # Metadata for Chinese dataset split
│   ├── dataset-split-en.json                   # Metadata for English dataset split
│   ├── dataset_dpo_training.json               # Dataset for DPO training
│   ├── skill-points-cn.json                    # Metadata for Chinese skill points
│   └── skill-points-en.json                    # Metadata for English skill points
├── README.md                                   # Introduction to this repository
└── requirements.txt                            # Python dependencies
```


## 1.3 Metadata Format of Data Item
- `id`: Unique identifier for the research problem (e.g., "en-0001").
- `title`: The title of the dissertation to which the research problem belongs.
- `core_research_question`: The main research problem addressed in the dissertation.
- `research_field`: The academic field pertaining to the research problem.
- `research_type`: The classification of research (e.g., "Application", "Theory").
- `year`: The year of the dissertation.

Example (from `dataset/dataset-split-en.json`):
```json
{
    "id": "en-0014",
    "title": "Graph Representation Learning-Based Recommender Systems",
    "core_research_question": "How to effectively integrate graph representation learning with recommender systems to enhance recommendation quality?",
    "research_field": "Computer Science",
    "research_type": "Application",
    "year": "2020"
}
```


## 1.4 Metadata Format of Skill Point
The metadata format of a skill point is as follows:

- `id`: Unique identifier for the skill point (e.g., "en-skill-point-0001").
- `problem_description`: A brief description of the research problem or challenge addressed by the skill point.
- `skill_point_name`: The name of the skill point.
- `skill_point_description`: A detailed description of the skill point.

Example (from `dataset/skill-points-en.json`):
```json
{
    "id": "en-skill-point-0001",
    "problem_description": "LLMs often hallucinate or generate unverified information, making them unreliable for scientific research where accuracy and traceability are paramount.",
    "skill_point_name": "Two-Pass Retrieval-Augmented Generation RAG with Usefulness Metric",
    "skill_point_description": "LmRaC employs a two-pass RAG approach. First, candidate text chunks are retrieved based on semantic similarity. Second, these candidates are explicitly filtered by the LLM for 'usefulness' in answering the question. This significantly improves answer quality and reduces hallucination by ensuring only the most relevant and high-utility information, traceable to paragraph-level citations, is presented."
}
```


## 1.5 Metadata Format of dpo training dataset
The metadata format of a DPO (Direct Preference Optimization) training dataset item is as follows:

- `id`: Unique identifier for the DPO training sample (e.g., "dpo-0001").
- `type`: Type of the sample (e.g., "reason").
- `core_research_question`: The core research problem.
- `roadmap`: The roadmap content (in Markdown format) used to solve the research problem to be evaluated.
- `system_prompt`: System prompt for evaluating the quality of the roadmap.
- `user_prompt`: User prompt containing the research problem and roadmap.
- `better_eval`: Better evaluation result, containing the score within `<eval_score>` tags and detailed analysis within `<eval_reason>` tags.
- `worse_eval`: Worse evaluation result, containing the score within `<eval_score>` tags and detailed analysis within `<eval_reason>` tags.


# 2 Quick Start
## 2.1 Environment Setup
Download this repository and navigate into the directory:
```bash
cd RoadMapper
```

Create your Conda environment with the following command:
```bash
conda create -n roadmapper python=3.11.5
```

Activate the environment:
```bash
conda activate roadmapper
```

Install the dependencies with the following command:
```bash
pip install -r requirements.txt
```


## 2.2 Setup Chroma
[Chroma](https://docs.trychroma.com/docs/overview/introduction) is a vector database used to store embedding data and retrieve relevant skill points for the `Knowledge Agent`.

1. Navigate to the Chroma directory:
```bash
cd chroma
```

2. Start the Chroma server:
```bash
bash start_chroma.sh
```

If the terminal displays the following text, the Chroma server is running successfully:
```bash
Saving data to: ./chroma-server
Connect to Chroma at: http://localhost:57778
Getting started guide: https://docs.trychroma.com/docs/overview/getting-started

OpenTelemetry is not enabled because it is missing from the config.
Listening on 0.0.0.0:57778
```

To proceed with the following steps, keep this terminal open to ensure the Chroma server continues running. You can open a new terminal to execute subsequent commands.

3. We provide a script (`chroma.py`) for basic operations on the Chroma server within this experiment.

Run the following command for help information:
```bash
python chroma.py
```

4. **To complete our experiments, you need to create the English and Chinese skill point collections:**
```bash
python chroma.py action=create-en
python chroma.py action=create-cn
```

The system will output the following text if a collection is successfully created (example for English collection):
```bash
Successfully added 2493 vectors to collection
```

To test the Chroma server, you can run the following commands:
```bash
python chroma.py action=test-en
python chroma.py action=test-cn
```

To delete all collections in the Chroma server, you can run the following command:
```bash
python chroma.py action=delete-all
```


## 2.3 Run the Pipeline
We use `./code/experiments/infer.py` for main experiment inference. Before executing the script, you need to configure the Config class:

**Common Configuration Items (usually need to be modified):**
- `dataset_file` (str): Dataset file path
- `model_name` (str): Model name for inference
- `model_client` (AsyncOpenAI): Model client configuration, requires setting `api_key` and `base_url`
- `evaluate_model_name` (str): Evaluate agent model name
- `evaluate_model_client` (AsyncOpenAI): Evaluate model client configuration, requires setting `api_key` and `base_url`
- `rpm` (int): Requests per minute (Rate Per Minute), used to control API call frequency
- `process_count` (int): Number of data items to process, set to `-1` to process all data
- `infer_mode` (str): Inference mode, optional values:
  - `"roadmapper"`: Use the complete RoadMapper system (default)
  - `"direct"`: Use direct prompting baseline
  - `"force_n_round"`: Force N rounds iteration

**RoadMapper Core Parameters:**
- `n_results` (int): Number of skill points retrieved by Knowledge Agent
- `max_iteration_count` (int): Maximum iteration rounds for RoadMapper, default is `5`
- `passing_score` (int): Passing score threshold for evaluation, default is `80`
- `max_retry_count_roadmapper` (int): Maximum retry count within RoadMapper, default is `10`

**Task Management Parameters:**
- `restore_from_dir` (str): Restore running state from specified directory (for resuming from checkpoint), empty string means no restoration needed
- `output_root_dir` (str): Output root directory
- `max_retry_count_per_item` (int): Maximum retry count per data item
- `shuffle` (bool): Whether to shuffle data order
- `max_processing_count` (int): Maximum concurrent processing count, set to `-1` for unlimited
- `item_process_detail_dir` (str): Directory for saving detailed processing results of each data item, default is `"item_process_detail"`

**ChromaDB Configuration:**
- `chroma_client` (chromadb.HttpClient): ChromaDB client
- `project_embedding_file_dir` (str): Core research question embedding file directory

Then return to the project root directory and execute:
```bash
cd ..
python code/experiments/infer.py
```


After running the pipeline, inference results will be stored in the `output_root_dir` directory. Each execution creates a timestamped directory with the following structure:
```bash
root/
├── YYYY-MM-DD-HH-MM-SS_process_count_N/
│   ├── merged_result.json
│   ├── item_json_detail/
│   │   ├── en-0001.json
│   │   ├── en-0002.json
│   │   └── ...
│   ├── item_process_detail/
│   │   ├── en-0001/
│   │   ├── en-0002/
│   │   └── ...
│   ├── item_fail_map.json
│   ├── process.log
│   ├── runtime_info.md
│   └── infer.py
```

**Key Output Files:**

1.  **`merged_result.json`**: The final merged inference results containing all successfully processed research questions. This file contains an array of results, with each item representing the inference result for a research problem.

2.  **`item_json_detail/`**: Contains individual JSON result files for each research question (e.g., `en-0001.json`), storing the complete inference result for that specific item.

3.  **`item_process_detail/`**: Contains detailed processing logs and intermediate files for each research question during the inference pipeline execution.

4.  **`item_fail_map.json`**: Records the failure count for each item that encountered errors during processing.

5.  **`process.log`**: Comprehensive execution log recording the pipeline execution process.

6.  **`runtime_info.md`**: Runtime statistics including success count, processing count, and item status information.

7.  **`infer.py`**: A copy of the inference script used for this execution, preserved for reproducibility.


## 2.4 Run the Evaluation

We use `./code/experiments/eval.py` for main experiment evaluation of inference outcome. Before executing the script, you need to configure the Config class:

**Common Configuration Items (usually need to be modified):**
- `dataset_file` (str): Dataset file path (should contain the inference results with roadmap field)
- `model_name` (str): Model name for evaluation
- `model_client` (AsyncOpenAI): Model client configuration, requires setting `api_key` and `base_url`
- `rpm` (int): Requests per minute (Rate Per Minute), used to control API call frequency
- `process_count` (int): Number of data items to process, set to `-1` to process all data

**Task Management Parameters:**
- `restore_from_dir` (str): Restore running state from specified directory (for resuming from checkpoint), empty string means no restoration needed
- `output_root_dir` (str): Output root directory
- `max_retry_count_per_item` (int): Maximum retry count per data item
- `shuffle` (bool): Whether to shuffle data order
- `max_processing_count` (int): Maximum concurrent processing count, set to `-1` for unlimited

**Evaluation Specific Parameters:**
- `golden_roadmap_dir` (str): Directory containing golden roadmap files (used as reference standard for evaluation)

Then, run the evaluation:
```bash
python code/experiments/eval.py
```

The evaluation results will be saved in the `output_root_dir` directory. Each run will create a timestamped directory with the following structure:

```bash
root/
├── YYYY-MM-DD-HH-MM-SS_process_count_N/
│   ├── merged_result.json
│   ├── item_json_detail/
│   │   ├── en-0001.json
│   │   ├── en-0002.json
│   │   └── ...
│   ├── evaluation_result.json
│   ├── process.log
│   ├── runtime_info.md
│   ├── item_fail_map.json
│   ├── dataset_remark.md
│   └── eval.py
```

**Key Output Files:**

1. **`merged_result.json`**: The final merged evaluation results containing all successfully processed research questions. Each item includes the original research problem metadata, the generated roadmap, and the evaluation results.

2. **`item_json_detail/`**: Contains individual JSON result files for each research question (e.g., `en-0001.json`), storing the complete evaluation result for that specific item, including:
   - Original research problem metadata (`id`, `title`, `core_research_question`, etc.)
   - Generated `roadmap` content
   - `evaluation_result` containing:
     - `step_score`: Key step representation score (0-100)
     - `logic_score`: Logical coherence score (0-100)
     - `degree_score`: Average out-degree score (0-100, DegreeScore in our paper)
     - `depth_score`: Average depth score (0-100, DepthScore in our paper)
     - `eval_reason`: Detailed evaluation reasoning

3. **`evaluation_result.json`**: Contains the average statistics of all evaluation metrics:
   - `average_step_score`: Average key step representation score
   - `average_logic_score`: Average logical coherence score
   - `average_degree_score`: Average out-degree score (DegreeScore in our paper)
   - `average_depth_score`: Average depth score (DepthScore in our paper)
   - `average_score`: Overall average score across all metrics

4. **`process.log`**: Comprehensive execution log recording the evaluation process.

5. **`runtime_info.md`**: Runtime statistics including success count, processing count, and item status information.

6. **`item_fail_map.json`**: Records the failure count for each item that encountered errors during processing.

7. **`dataset_remark.md`**: Records the dataset file path used for evaluation.

8. **`eval.py`**: A copy of the evaluation script used for this execution, preserved for reproducibility.