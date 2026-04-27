import time
import asyncio
from openai import AsyncOpenAI
import json
import os
import shutil
import chromadb
from .agents.init_agent import InitAgent
from .agents.knowledge_agent import KnowledgeAgent, CreateTechLib
from .agents.logic_critique_agent import LogicCritiqueAgent
from .agents.granularity_critique_agent import GranularityCritiqueAgent
from .agents.revise_agent import ReviseAgent
from .agents.evaluate_agent import EvaluateAgent
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

    # =========================Part 2, augment roadmap with knowledge=========================
    tech_lib = None
    tech_lib_model_response = None
    for retry in range(max_retry_count):
        try:
            tech_lib, tech_lib_model_response = await CreateTechLib(client, model, project, chroma_client, collection_name, project_embedding_file_path, n_results, split_language)
            if not tech_lib:
                if retry < max_retry_count - 1:
                    print(f"[Retry {retry + 1}/{max_retry_count}] Knowledge point repository generation failed, retrying...")
                    continue
                else:
                    raise Exception("[Error] Knowledge point repository generation failed after all retries")
            
            # Success
            break
        except Exception as e:
            if retry < max_retry_count - 1:
                print(f"[Retry {retry + 1}/{max_retry_count}] Exception occurred: {str(e)}, retrying...")
                continue
            else:
                raise

    SaveFileUtil(os.path.join(project_outcome_root_dir, "02-tech_lib.json"), json.dumps(tech_lib, ensure_ascii=False, indent=4))
    SaveFileUtil(os.path.join(project_outcome_root_dir, "02-tech_lib_model_response.json"), json.dumps(tech_lib_model_response, ensure_ascii=False, indent=4))

    knowledge_augmented_roadmap = None
    knowledge_augmented_roadmap_model_response = None
    for retry in range(max_retry_count):
        try:
            knowledge_augmented_roadmap, knowledge_augmented_roadmap_model_response = await KnowledgeAgent(client, model, project, initial_roadmap, tech_lib, split_language)
            if not knowledge_augmented_roadmap:
                if retry < max_retry_count - 1:
                    print(f"[Retry {retry + 1}/{max_retry_count}] Knowledge augmented roadmap generation failed, retrying...")
                    continue
                else:
                    raise Exception("[Error] Knowledge augmented roadmap generation failed after all retries")

            knowledge_augmented_roadmap, is_any_wrong_line = FormatRectify(knowledge_augmented_roadmap)
            if is_any_wrong_line:
                if retry < max_retry_count - 1:
                    SaveFileUtil(os.path.join(project_outcome_root_dir, f"03-knowledge_augmented_roadmap_error_retry{retry+1}.md"), knowledge_augmented_roadmap)
                    print(f"[Retry {retry + 1}/{max_retry_count}] Knowledge augmented roadmap has format errors, retrying...")
                    continue
                else:
                    SaveFileUtil(os.path.join(project_outcome_root_dir, "03-knowledge_augmented_roadmap_error.md"), knowledge_augmented_roadmap)
                    raise Exception("[Error] Knowledge augmented roadmap has format errors after all retries")
            
            if not CheckIndexValidity(knowledge_augmented_roadmap):
                if retry < max_retry_count - 1:
                    SaveFileUtil(os.path.join(project_outcome_root_dir, f"03-knowledge_augmented_roadmap_index_error_retry{retry+1}.md"), knowledge_augmented_roadmap)
                    print(f"[Retry {retry + 1}/{max_retry_count}] Knowledge augmented roadmap has index errors, retrying...")
                    continue
                else:
                    SaveFileUtil(os.path.join(project_outcome_root_dir, "03-knowledge_augmented_roadmap_index_error.md"), knowledge_augmented_roadmap)
                    raise Exception("[Error] Knowledge augmented roadmap has index errors after all retries")
                
            # Success
            break
        except Exception as e:
            if retry < max_retry_count - 1:
                print(f"[Retry {retry + 1}/{max_retry_count}] Exception occurred: {str(e)}, retrying...")
                continue
            else:
                raise

    SaveFileUtil(os.path.join(project_outcome_root_dir, "03-knowledge_augmented_roadmap.md"), knowledge_augmented_roadmap)
    SaveFileUtil(os.path.join(project_outcome_root_dir, "03-knowledge_augmented_roadmap_model_response.json"), json.dumps(knowledge_augmented_roadmap_model_response, ensure_ascii=False, indent=4))

    # =========================Part 3, iteratively revise roadmap=========================
    roadmap = knowledge_augmented_roadmap
    for i in range(max_iteration_count):
        # Check the logic dependency and granularity of the roadmap
        logic_critique_result = None
        logic_critique_model_response = None
        granularity_critique_result = None
        granularity_critique_model_response = None
        
        for retry in range(max_retry_count):
            try:
                logic_critique_result, logic_critique_model_response = await LogicCritiqueAgent(client, model, project, roadmap)
                granularity_critique_result, granularity_critique_model_response = await GranularityCritiqueAgent(client, model, project, roadmap)

                if not logic_critique_result or not granularity_critique_result:
                    if retry < max_retry_count - 1:
                        print(f"[Retry {retry + 1}/{max_retry_count}] The {i+1}th iteration, logic dependency or granularity check failed, retrying...")
                        continue
                    else:
                        raise Exception(f"[Error] The {i+1}th iteration, logic dependency or granularity check failed after all retries")
                
                # Success
                break
            except Exception as e:
                if retry < max_retry_count - 1:
                    print(f"[Retry {retry + 1}/{max_retry_count}] The {i+1}th iteration, Exception occurred: {str(e)}, retrying...")
                    continue
                else:
                    raise

        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-logic_critique_result.md"), logic_critique_result)
        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-granularity_critique_result.md"), granularity_critique_result)
        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-logic_critique_model_response.json"), json.dumps(logic_critique_model_response, ensure_ascii=False, indent=4))
        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-granularity_critique_model_response.json"), json.dumps(granularity_critique_model_response, ensure_ascii=False, indent=4))

        # Revise the roadmap based on the critique results
        revised_roadmap = None
        taken_measures = None
        revised_roadmap_model_response = None
        
        for retry in range(max_retry_count):
            try:
                revised_roadmap, taken_measures, revised_roadmap_model_response = await ReviseAgent(client, model, project, roadmap, split_language, logic_critique_result, granularity_critique_result)

                if not revised_roadmap:
                    if retry < max_retry_count - 1:
                        print(f"[Retry {retry + 1}/{max_retry_count}] The {i+1}th iteration, roadmap revision failed, retrying...")
                        continue
                    else:
                        raise Exception(f"[Error] The {i+1}th iteration, roadmap revision failed after all retries")

                revised_roadmap, is_any_wrong_line = FormatRectify(revised_roadmap)
                if is_any_wrong_line:
                    if retry < max_retry_count - 1:
                        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-revised_roadmap_error_retry{retry+1}.md"), revised_roadmap)
                        print(f"[Retry {retry + 1}/{max_retry_count}] The {i+1}th iteration, the revised roadmap has format errors, retrying...")
                        continue
                    else:
                        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-revised_roadmap_error.md"), revised_roadmap)
                        raise Exception(f"[Error] The {i+1}th iteration, the revised roadmap has format errors after all retries")
                
                if not CheckIndexValidity(revised_roadmap):
                    if retry < max_retry_count - 1:
                        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-revised_roadmap_index_error_retry{retry+1}.md"), revised_roadmap)
                        print(f"[Retry {retry + 1}/{max_retry_count}] The {i+1}th iteration, the revised roadmap has index errors, retrying...")
                        continue
                    else:
                        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-revised_roadmap_index_error.md"), revised_roadmap)
                        raise Exception(f"[Error] The {i+1}th iteration, the revised roadmap has index errors after all retries")
                
                # Success
                break
            except Exception as e:
                if retry < max_retry_count - 1:
                    print(f"[Retry {retry + 1}/{max_retry_count}] The {i+1}th iteration, Exception occurred: {str(e)}, retrying...")
                    continue
                else:
                    raise

        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-revised_roadmap.md"), revised_roadmap)
        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-taken_measures.md"), taken_measures)
        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-revised_roadmap_model_response.json"), json.dumps(revised_roadmap_model_response, ensure_ascii=False, indent=4))

        # Evaluate the quality of the revised roadmap
        evaluation_score = None
        evaluation_reason = None
        evaluation_model_response = None
        
        for retry in range(max_retry_count):
            try:
                evaluation_score, evaluation_reason, evaluation_model_response = await EvaluateAgent(
                    evaluate_model_client,
                    evaluate_model_name,
                    project,
                    revised_roadmap
                )

                if evaluation_score is None:
                    if retry < max_retry_count - 1:
                        print(f"[Retry {retry + 1}/{max_retry_count}] The {i+1}th iteration, evaluation failed, retrying...")
                        continue
                    else:
                        raise Exception(f"[Error] The {i+1}th iteration, evaluation failed after all retries")
                
                # Success
                break
            except Exception as e:
                if retry < max_retry_count - 1:
                    print(f"[Retry {retry + 1}/{max_retry_count}] The {i+1}th iteration, Exception occurred: {str(e)}, retrying...")
                    continue
                else:
                    raise

        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-evaluation_score.json"), json.dumps({"score": evaluation_score, "eval_reason": evaluation_reason}, ensure_ascii=False, indent=4))
        SaveFileUtil(os.path.join(project_outcome_root_dir, f"04-{i+1}-evaluation_model_response.json"), json.dumps(evaluation_model_response, ensure_ascii=False, indent=4))

        roadmap = revised_roadmap
        if evaluation_score >= passing_score:
            break

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    SaveFileUtil(os.path.join(project_outcome_root_dir, "05-final_roadmap.md"), roadmap)
    return roadmap

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
