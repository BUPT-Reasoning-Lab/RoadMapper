import time
import asyncio
from openai import AsyncOpenAI
import json
import os
import shutil
import chromadb
from .agents.init_agent import InitAgent
from .agents.utils import SaveFileUtil, FormatRectify, CheckIndexValidity


async def main_async(
    client: AsyncOpenAI,
    model: str,
    project: str,
    chroma_client: chromadb.HttpClient,
    collection_name: str,
    project_embedding_file_path: str,
    n_results: int,
    max_iteration_count: int,
    passing_score: int,
    project_outcome_root_dir: str,
    split_language: str="English",
    max_retry_count: int=3,
    evaluate_model_client: AsyncOpenAI=None,
    evaluate_model_name: str=None,
):
    """Main asynchronous pipeline for roadmap generation and improvement"""
    print(f"Start processing project: {project}")
    print(f"Output directory: {project_outcome_root_dir}")
    start_time = time.time()
    if os.path.exists(project_outcome_root_dir):
        shutil.rmtree(project_outcome_root_dir)
    os.makedirs(project_outcome_root_dir, exist_ok=True)

    # =========================Part 1, generate initial roadmap=========================
    initial_roadmap = None
    initial_roadmap_model_response = None
    for retry in range(max_retry_count):
        try:
            initial_roadmap, initial_roadmap_model_response = await InitAgent(client, model, project, split_language)
            if not initial_roadmap:
                if retry < max_retry_count - 1:
                    print(f"[Retry {retry + 1}/{max_retry_count}] Initial roadmap generation failed, retrying...")
                    continue
                else:
                    raise Exception("[Error] Initial roadmap generation failed after all retries")

            initial_roadmap, is_any_wrong_line = FormatRectify(initial_roadmap)
            if is_any_wrong_line:
                if retry < max_retry_count - 1:
                    SaveFileUtil(os.path.join(project_outcome_root_dir, f"01-initial_roadmap_error_retry{retry+1}.md"), initial_roadmap)
                    print(f"[Retry {retry + 1}/{max_retry_count}] Initial roadmap has format errors, retrying...")
                    continue
                else:
                    SaveFileUtil(os.path.join(project_outcome_root_dir, "01-initial_roadmap_error.md"), initial_roadmap)
                    raise Exception("[Error] Initial roadmap has format errors after all retries")
            
            if not CheckIndexValidity(initial_roadmap):
                if retry < max_retry_count - 1:
                    SaveFileUtil(os.path.join(project_outcome_root_dir, f"01-initial_roadmap_index_error_retry{retry+1}.md"), initial_roadmap)
                    print(f"[Retry {retry + 1}/{max_retry_count}] Initial roadmap has index errors, retrying...")
                    continue
                else:
                    SaveFileUtil(os.path.join(project_outcome_root_dir, "01-initial_roadmap_index_error.md"), initial_roadmap)
                    raise Exception("[Error] Initial roadmap has index errors after all retries")
            # Success
            break
        except Exception as e:
            if retry < max_retry_count - 1:
                print(f"[Retry {retry + 1}/{max_retry_count}] Exception occurred: {str(e)}, retrying...")
                continue
            else:
                raise

    SaveFileUtil(os.path.join(project_outcome_root_dir, "01-initial_roadmap.md"), initial_roadmap)
    SaveFileUtil(os.path.join(project_outcome_root_dir, "01-initial_roadmap_model_response.json"), json.dumps(initial_roadmap_model_response, ensure_ascii=False, indent=4))

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    SaveFileUtil(os.path.join(project_outcome_root_dir, "05-final_roadmap.md"), initial_roadmap)
    return initial_roadmap

def main(
    client: AsyncOpenAI,
    model: str,
    project: str,
    chroma_client: chromadb.HttpClient,
    collection_name: str,
    project_embedding_file_path: str,
    n_results: int,
    max_iteration_count: int,
    passing_score: int,
    project_outcome_root_dir: str,
    split_language: str="English",
    max_retry_count: int=3,
    evaluate_model_client: AsyncOpenAI=None,
    evaluate_model_name: str=None,
):
    """Synchronous wrapper for the main asynchronous pipeline"""
    return asyncio.run(
        main_async(
            client,
            model,
            project,
            chroma_client,
            collection_name,
            project_embedding_file_path,
            n_results,
            max_iteration_count,
            passing_score,
            project_outcome_root_dir,
            split_language,
            max_retry_count,
            evaluate_model_client,
            evaluate_model_name,
        )
    )
