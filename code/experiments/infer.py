import asyncio
import json
import os
import sys
import logging
import random
import shutil
import argparse
from datetime import datetime
from collections import deque
from typing import Any
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
import chromadb

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from roadmapper import roadmapper_retry as roadmapper_retry_main
from roadmapper import force_n_round as force_n_round
from roadmapper import direct_prompting as direct_prompting


ITEM_ID_KEY = "id"
PROCESS_RESULT_KEY = "process_result"
IS_PROCESS_SUCCESS_KEY = "is_success"
ITEM_JSON_DETAIL_DIR = "item_json_detail"
PROCESS_LOG_FILE_NAME = "process.log"
RECOVER_REMARK_FILE_NAME = "recover_remark.md"
PROCESS_MERGED_RESULT_FILE_NAME = "merged_result.json"
ITEM_FAIL_MAP_FILE_NAME = "item_fail_map.json"
RUNTIME_INFO_FILE_NAME = "runtime_info.md"


class Config:
    """Configuration class"""

    def __init__(self):
        self.restore_from_dir = ""
        self.output_root_dir = "output"
        self.max_retry_count_per_item = 10
        self.shuffle = False
        self.max_processing_count = 300
        self.chroma_client = chromadb.HttpClient(host="127.0.0.1", port=57778)
        self.project_embedding_file_dir = "../data/core-research-question-embeddings"
        self.n_results = 30
        self.max_iteration_count = 5
        self.passing_score = 80
        self.max_retry_count_roadmapper = 10
        self.item_process_detail_dir = "item_process_detail"
        
        # usually change items
        self.dataset_file = "../data/dataset-split-en.json"
        self.model_name = "meta-llama/llama-3.3-70b-instruct"
        self.model_client = AsyncOpenAI(
            api_key="",
            base_url=""
        )
        self.evaluate_model_name = "gpt-4o-mini"
        self.evaluate_model_client = AsyncOpenAI(
            api_key="",
            base_url=""
        )
        self.rpm = 200
        self.process_count = 1
        self.infer_mode = "roadmapper"


class TaskManager:
    """Task manager"""

    async def process_one_item(self, item: Any):
        """Process a data item using the specified worker"""
        raw_item = item.copy()
        item_id = item[ITEM_ID_KEY]
        self.processing_items.append(item_id)
        try:
            # ============================ Processing logic for each data item starts ============================
            dataset_basename = os.path.basename(self.config.dataset_file)
            if "-en" in dataset_basename:
                split_language = "English"
                collection_name = "skill_points_en"
            elif "-cn" in dataset_basename:
                split_language = "Chinese"
                collection_name = "skill_points_cn"
            else:
                split_language = "English"
                collection_name = "skill_points_en"
                
            
            main_func_async = None
            if self.config.infer_mode == "roadmapper":
                main_func_async = roadmapper_retry_main.main_async
            elif self.config.infer_mode == "direct":
                main_func_async = direct_prompting.main_async
            elif self.config.infer_mode == "force_n_round":
                main_func_async = force_n_round.main_async
            else:
                raise ValueError(f"Invalid infer mode: {self.config.infer_mode}")
            
            roadmap = await main_func_async(
                client=self.config.model_client,
                model=self.config.model_name,
                project=item["core_research_question"],
                chroma_client=self.config.chroma_client,
                collection_name=collection_name,
                project_embedding_file_path=os.path.join(self.config.project_embedding_file_dir, f"{item['id']}.json"),
                n_results=self.config.n_results,
                max_iteration_count=self.config.max_iteration_count,
                passing_score=self.config.passing_score,
                project_outcome_root_dir=os.path.join(self.current_run_output_dir, self.config.item_process_detail_dir, item["id"]),
                split_language=split_language,
                max_retry_count=self.config.max_retry_count_roadmapper,
                evaluate_model_client=self.config.evaluate_model_client,
                evaluate_model_name=self.config.evaluate_model_name,
            )

            item["roadmap"] = roadmap
            # ============================ Processing logic for each data item ends ============================

            self.success_items.append(item_id)
            item[PROCESS_RESULT_KEY] = {IS_PROCESS_SUCCESS_KEY: True}
            self.save_item_result(item)
        except Exception as e:
            item[PROCESS_RESULT_KEY] = {IS_PROCESS_SUCCESS_KEY: False, "error": str(e)}
            self.save_item_result(item)
            self.item_fail_map[item_id] += 1
            if self.item_fail_map[item_id] > self.config.max_retry_count_per_item:
                self.logger.error(
                    f"Item {item_id} failed after {self.config.max_retry_count_per_item} retries, giving up."
                )
                self.stop_schedule = True
                return
            self.data_queue.append(raw_item)
            self.logger.error(f"Item {item_id} failed with error: {e}")
        finally:
            if item_id in self.processing_items:
                self.processing_items.remove(item_id)

    def __init__(
        self,
        config: Config,
    ):
        self.config = config
        self.dataset = None
        self.data_queue = deque[Any]()
        self.success_items = []
        self.processing_items = []
        self.item_fail_map = {}
        self.stop_schedule = False
        self.is_restore_from_dir = bool(self.config.restore_from_dir)
        self.active_tasks = []

        with open(self.config.dataset_file, "r") as f:
            self.dataset = json.load(f)
            self.dataset = (
                self.dataset[: self.config.process_count]
                if self.config.process_count > 0
                else self.dataset
            )
            if self.config.shuffle:
                random.shuffle(self.dataset)
            self.total_count = len(self.dataset)

        self.current_run_output_dir = self.create_output_dir(self.config)

        if self.is_restore_from_dir:
            self.restore_from_dir()
        else:
            for item in self.dataset:
                self.data_queue.append(item)
                self.item_fail_map[item[ITEM_ID_KEY]] = 0

        self.logger = self.setup_logging()

    def restore_from_dir(self):
        """Restore running state from existing output directory"""
        restore_dir = self.config.restore_from_dir
        if not os.path.isdir(restore_dir):
            raise ValueError(f"restore_from_dir path does not exist: {restore_dir}")

        shutil.copytree(restore_dir, self.current_run_output_dir, dirs_exist_ok=True)

        # Read completed results
        processed_item_ids = set()
        item_json_detail_dir = os.path.join(
            self.current_run_output_dir, ITEM_JSON_DETAIL_DIR
        )
        if os.path.isdir(item_json_detail_dir):
            item_json_detail_files = [
                fname
                for fname in os.listdir(item_json_detail_dir)
                if fname.endswith(".json")
            ]
            for fname in item_json_detail_files:
                fpath = os.path.join(item_json_detail_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    item = json.load(f)
                item_id = item.get(ITEM_ID_KEY)
                is_success = item.get(PROCESS_RESULT_KEY, {}).get(
                    IS_PROCESS_SUCCESS_KEY
                )
                if item_id is not None and is_success:
                    processed_item_ids.add(item_id)
        else:
            self.logger.warning(
                f"Item JSON detail directory not found, treating as no completed data, directory: {item_json_detail_dir}"
            )

        # Generate queue and success list based on dataset
        for item in self.dataset:
            item_id = item.get(ITEM_ID_KEY)
            if item_id in processed_item_ids:
                self.success_items.append(item_id)
            else:
                self.data_queue.append(item)
                self.item_fail_map[item_id] = 0

        # Record recovery information for tracking
        remark_path = os.path.join(
            self.current_run_output_dir, RECOVER_REMARK_FILE_NAME
        )
        with open(remark_path, "w", encoding="utf-8") as f:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            left_ids = [item.get(ITEM_ID_KEY) for item in self.data_queue]
            f.write(
                f"Time: {now_str}\n"
                f"Restored from: {restore_dir}\n"
                f"New output dir: {self.current_run_output_dir}\n"
                f"Completed items count: {len(self.success_items)}\n"
                f"Remaining items count: {len(self.data_queue)}\n"
                f"Remaining items: {', '.join(left_ids) if left_ids else '(empty)'}\n"
            )

    def can_enqueue(self) -> bool:
        """Check if items can be enqueued"""
        return (
            len(self.processing_items) <= self.config.max_processing_count
            if self.config.max_processing_count > 0
            else True
        )

    def setup_logging(self):
        """Setup logging"""
        # Set logging level for openai and httpx to WARNING to avoid excessive HTTP request logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

        log_file_path = os.path.join(self.current_run_output_dir, PROCESS_LOG_FILE_NAME)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file_path, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        return logging.getLogger(__name__)

    def create_output_dir(self, config: Config) -> str:
        """Create output directory named with timestamp and processing data count"""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Extract the last part of model name
        model_name_part = config.model_name.split("/")[-1]
        output_dir = os.path.join(
            config.output_root_dir,
            config.dataset_file.split("/")[-1].split(".")[0],
            model_name_part,
            f"{timestamp}_process_count_{self.total_count}",
        )
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, ITEM_JSON_DETAIL_DIR), exist_ok=True)
        return output_dir

    def save_item_result(self, item: Any):
        """Save result"""
        with open(
            os.path.join(
                self.current_run_output_dir,
                ITEM_JSON_DETAIL_DIR,
                f"{item[ITEM_ID_KEY]}.json",
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(item, f, indent=4, ensure_ascii=False)

    def merge_all_item_results(self):
        """After all items are processed, merge all item results into a json list file sorted by filename"""
        item_json_detail_dir = os.path.join(
            self.current_run_output_dir, ITEM_JSON_DETAIL_DIR
        )
        # Get all json filenames and sort by filename
        file_list = sorted(
            [f for f in os.listdir(item_json_detail_dir) if f.endswith(".json")]
        )
        merged_items = []
        for fname in file_list:
            fpath = os.path.join(item_json_detail_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                item = json.load(f)
                merged_items.append(item)
        # Save merged results
        merged_file_path = os.path.join(
            self.current_run_output_dir, PROCESS_MERGED_RESULT_FILE_NAME
        )
        with open(merged_file_path, "w", encoding="utf-8") as f:
            json.dump(merged_items, f, indent=4, ensure_ascii=False)

    def save_runtime_info(self):
        """Save current runtime information to file"""
        # Save item_fail_map to file
        item_fail_map_file_path = os.path.join(
            self.current_run_output_dir, ITEM_FAIL_MAP_FILE_NAME
        )
        with open(item_fail_map_file_path, "w", encoding="utf-8") as f:
            json.dump(self.item_fail_map, f, indent=4, ensure_ascii=False)

        queue_item_ids = [item[ITEM_ID_KEY] for item in self.data_queue]

        # Find all items with failure count equal to maximum count
        max_failed_items = "(empty)"
        max_fail_count = 0
        if self.item_fail_map:
            max_fail_count = max(self.item_fail_map.values())
            if max_fail_count > 0:
                max_failed_item_list = [
                    f"{item_id}"
                    for item_id, fail_count in self.item_fail_map.items()
                    if fail_count == max_fail_count
                ]
                max_failed_items = ", ".join(max_failed_item_list)

        detail_info = f"**Success count**: {len(self.success_items)};\n"
        detail_info += f"**Processing count**: {len(self.processing_items)};\n"
        detail_info += f"**Total count**: {self.total_count};\n\n"
        detail_info += f"**Success items**: {', '.join(map(str, self.success_items)) if self.success_items else '(empty)'}\n\n"
        detail_info += f"**Processing items**: {', '.join(map(str, self.processing_items)) if self.processing_items else '(empty)'}\n\n"
        detail_info += f"**Queue items**: {', '.join(queue_item_ids) if queue_item_ids else '(empty)'}\n\n"
        detail_info += f"**Max Failed Count**: {max_fail_count}\n\n"
        detail_info += f"**Max Failed Items**: {max_failed_items}\n\n"
        runtime_info_file_path = os.path.join(
            self.current_run_output_dir, RUNTIME_INFO_FILE_NAME
        )
        with open(runtime_info_file_path, "w", encoding="utf-8") as f:
            f.write(detail_info)

    def copy_current_file(self):
        """Copy current Python file to output directory"""
        current_file_path = __file__
        copied_file_path = os.path.join(
            self.current_run_output_dir, os.path.basename(current_file_path)
        )
        shutil.copy2(current_file_path, copied_file_path)
        self.logger.info(f"Python file copied to {copied_file_path}")

    async def main_schedule(self):
        """Run task manager"""
        self.logger.info("Task manager started.")
        if self.is_restore_from_dir:
            self.logger.info(f"Restored from: {self.config.restore_from_dir}")
            self.logger.info(
                f"Completed: {len(self.success_items)}, Remaining: {len(self.data_queue)}"
            )

        self.copy_current_file()

        with tqdm(total=self.total_count, initial=len(self.success_items)) as pbar:
            last_success = len(self.success_items)
            while len(self.success_items) < self.total_count and not self.stop_schedule:
                await asyncio.sleep(60 / self.config.rpm)

                if self.data_queue and self.can_enqueue():
                    self.active_tasks.append(
                        asyncio.create_task(
                            self.process_one_item(self.data_queue.popleft())
                        )
                    )

                self.active_tasks = [t for t in self.active_tasks if not t.done()]
                pbar.update(len(self.success_items) - last_success)
                last_success = len(self.success_items)
                self.save_runtime_info()

        self.active_tasks = [t for t in self.active_tasks if not t.done()]
        if self.active_tasks:
            self.logger.info(
                f"Waiting for {len(self.active_tasks)} active tasks to complete..."
            )
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
            self.logger.info("All active tasks completed.")

        self.merge_all_item_results()
        self.save_runtime_info()
        self.logger.info(
            f"Task manager finished. Success: {len(self.success_items)}/{self.total_count}"
        )
        self.logger.info(f"Output directory: {self.current_run_output_dir}")


async def main():
    """Main function, override dataset file and model name via command line only"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    config = Config()
    if args.dataset_file:
        config.dataset_file = args.dataset_file
    if args.model_name:
        config.model_name = args.model_name

    manager = TaskManager(config)
    await manager.main_schedule()


if __name__ == "__main__":
    asyncio.run(main())
