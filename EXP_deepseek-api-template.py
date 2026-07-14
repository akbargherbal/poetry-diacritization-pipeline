# =============================================================================
# DeepSeek API — Parameter Reference (deepseek-v4-pro, OpenAI-compatible)
# NOTE: frequency_penalty and presence_penalty are deprecated — do not use.
# NOTE: top_p has no effect at temperature ≤ 0.2 (mass already concentrated).
#
#   Use case                         temperature    top_p
#   ─────────────────────────────── ──────────── ───────
#   Structured output / JSON                0.2      0.9
#   Constrained creative (Arudi)            0.4      0.9   ← current
#   General / mixed                         0.6      0.9
#   Open-ended creative / brainstorm        0.8      0.95
#   Reasoning model (thinking mode)*        n/a      n/a
#
#   * deepseek-v4-pro thinking mode: temperature, top_p have no effect.
#     Switch model to "deepseek-v4-pro" with stream=False; reasoning via
#     response.choices[0].message.reasoning_content.
# =============================================================================

import os
import sys
import time
import logging
import threading
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import OpenAI
import pandas as pd

# User-defined constants
GPT4_MODEL = "deepseek-v4-pro"
MAX_TOKENS = 64000
PROMPT_INDEX_COLUMN = "PROMPT_ID"
PROMPT_COLUMN = "PROMPT"
TEMPERATURE = 0.6  #
TOP_P = 0.9  # Clips low-probability tail tokens at this temperature — minor but meaningful for Arabic morphology
MAX_WORKERS = 6  # Number of concurrent threads
REQUESTS_PER_MINUTE = 6  # Rate limit
MAX_RETRY_ATTEMPTS = 4  # Maximum number of retry passes
REASONING_EFFORT = "high"  # Options: "high", "max"

# Additional columns to include in results
ADDITIONAL_COLUMNS = sorted(set("PATH	MODULE_NO".split()))

# Time-stamped output directory
TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_RESULT = f"results_{TIME_STAMP}"
LogFileName = "LOG.log"
PKL_FILE_NAME = "RESULTS.pkl"

# Create directory to save results if it doesn't exist
os.makedirs(DIR_RESULT, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(DIR_RESULT, LogFileName)),
        logging.StreamHandler(),
    ],
)


class SlidingWindowRateLimiter:
    """Thread-safe rate limiter using sliding window algorithm without lock-sleep bottleneck."""

    def __init__(self, max_requests, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()
        self.lock = threading.Lock()

    def acquire(self):
        """Wait until a request can be made without blocking other threads."""
        while True:
            sleep_time = 0
            with self.lock:
                current_time = time.time()

                # Clean expired timestamps
                while (
                    self.request_times
                    and self.request_times[0] <= current_time - self.time_window
                ):
                    self.request_times.popleft()

                if len(self.request_times) < self.max_requests:
                    # Slot available, reserve it
                    self.request_times.append(current_time)
                    return
                else:
                    # Calculate sleep required
                    sleep_time = (
                        self.time_window - (current_time - self.request_times[0]) + 0.1
                    )

            # Sleep outside the lock so other threads can evaluate concurrently
            if sleep_time > 0:
                logging.debug(
                    f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds"
                )
                time.sleep(sleep_time)


class ResultsManager:
    """Thread-safe manager for saving results incrementally."""

    def __init__(self, output_dir, pkl_filename):
        self.output_dir = output_dir
        self.pkl_filename = pkl_filename
        self.results_list = []
        self.lock = threading.Lock()
        self.processed_ids = set()

        # Load existing results if any
        pkl_path = os.path.join(output_dir, pkl_filename)
        if os.path.exists(pkl_path):
            try:
                df_existing = pd.read_pickle(pkl_path)
                self.results_list = df_existing.to_dict("records")
                self.processed_ids = set(df_existing[PROMPT_INDEX_COLUMN].values)
                logging.info(
                    f"Loaded {len(self.results_list)} existing results from {pkl_path}"
                )
            except Exception as e:
                logging.warning(f"Could not load existing results: {str(e)}")

    def add_result(self, result_dict):
        """Add a result and save incrementally."""
        with self.lock:
            self.results_list.append(result_dict)
            self.processed_ids.add(result_dict["PROMPT_ID"])

            # Save incrementally
            try:
                df_result = pd.DataFrame(self.results_list)
                pkl_path = os.path.join(self.output_dir, self.pkl_filename)
                df_result.to_pickle(pkl_path, protocol=4)
                logging.debug(f"Saved result for prompt ID: {result_dict['PROMPT_ID']}")
            except Exception as e:
                logging.error(f"Error saving result: {str(e)}")

    def is_processed(self, prompt_id):
        """Check if a prompt has already been processed."""
        with self.lock:
            return prompt_id in self.processed_ids

    def get_results_count(self):
        """Get the number of results."""
        with self.lock:
            return len(self.results_list)


def setup_openai_api():
    """Set up the OpenAI API key."""
    logging.info("Setting up DeepSeek API connection...")

    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("DEEPSEEK_API_KEY not found in environment variables.")
            api_key = input("Paste your DeepSeek API key: ").strip()
    except Exception as e:
        logging.error(f"Error getting API key: {str(e)}")
        api_key = input("Paste your DeepSeek API key: ").strip()

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        client.models.list()
        logging.info("DeepSeek API authentication successful.")
    except openai.AuthenticationError:
        logging.error("Authentication failed. Please check your API key.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while setting up the DeepSeek API: {str(e)}")
        sys.exit(1)

    return client


def generate_response(client, prompt_text, prompt_id, rate_limiter, system_prompt):
    """Generate a response using DeepSeek API with explicit reasoning controls."""
    rate_limiter.acquire()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})

    try:
        # Note: In thinking mode, temperature and top_p are ignored by the model,
        # but you can leave them or omit them without causing errors.
        response = client.chat.completions.create(
            model=GPT4_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            reasoning_effort=REASONING_EFFORT,
            extra_body={"thinking": {"type": "enabled"}},
        )

        result_content = response.choices[0].message.content

        # Safe extraction of the reasoning content trace
        reasoning_content = getattr(
            response.choices[0].message, "reasoning_content", None
        )

        # Save individual result files
        prefix = str(prompt_id).zfill(4)

        # 1. Save final answer
        result_file_name = f"{prefix}_result.txt"
        with open(
            os.path.join(DIR_RESULT, result_file_name), "w", encoding="utf-8"
        ) as f:
            f.write(result_content or "")

        # 2. Save reasoning trace if it was returned
        if reasoning_content:
            reasoning_file_name = f"{prefix}_reasoning.txt"
            with open(
                os.path.join(DIR_RESULT, reasoning_file_name), "w", encoding="utf-8"
            ) as f:
                f.write(reasoning_content)

        logging.info(f"Successfully processed prompt ID: {prompt_id}")
        return result_content, reasoning_content, response

    except Exception as e:
        logging.error(f"Error generating response for prompt ID {prompt_id}: {str(e)}")
        return None, None, None


def process_single_prompt(client, prompt_data, rate_limiter, results_manager, df):
    """
    Process a single prompt and save results.

    Args:
        client: OpenAI client instance
        prompt_data: Dictionary containing prompt_id, prompt_text, and system_prompt
        rate_limiter: RateLimiter instance
        results_manager: ResultsManager instance
        df: Original DataFrame for additional columns

    Returns:
        tuple: (prompt_id, success, error_message)
    """
    prompt_id = prompt_data["prompt_id"]
    prompt_text = prompt_data["prompt_text"]
    system_prompt = prompt_data["system_prompt"]

    result_content, reasoning_content, response = generate_response(
        client, prompt_text, prompt_id, rate_limiter, system_prompt
    )

    if result_content is not None:
        result_dict = {
            "PROMPT_ID": prompt_id,
            "RESULT": result_content,
            "REASONING": reasoning_content,  # Now capturing how DeepSeek reasons
            "RESPONSE": response,
        }

        # Add additional columns
        for col in ADDITIONAL_COLUMNS:
            if col in df.columns:
                result_dict[col] = df.loc[
                    df[PROMPT_INDEX_COLUMN] == prompt_id, col
                ].values[0]
            else:
                logging.warning(f"Column '{col}' not found in the input DataFrame")

        results_manager.add_result(result_dict)
        return prompt_id, True, None
    else:
        return prompt_id, False, "API call failed"


def process_prompts(client, path_to_prompts):
    """Process prompts with concurrent execution and retry logic."""

    # Load DataFrame
    try:
        df = pd.read_pickle(path_to_prompts)
        logging.info(f"Columns in the dataframe: {list(df.columns)}")
        print(f"Columns in the dataframe: {list(df.columns)}")

        if PROMPT_COLUMN not in df.columns:
            raise KeyError(f"'{PROMPT_COLUMN}' column not found in the input DataFrame")

        if PROMPT_INDEX_COLUMN not in df.columns:
            raise KeyError(
                f"'{PROMPT_INDEX_COLUMN}' column not found in the input DataFrame"
            )

    except FileNotFoundError:
        logging.error(f"Input file {path_to_prompts} not found.")
        return
    except KeyError as e:
        logging.error(f"Column error: {str(e)}")
        return
    except Exception as e:
        logging.error(f"Error reading input file: {str(e)}")
        return

    # Check for 'SYSTEM_PROMPT' column or load from 'SYSTEM_PROMPT.md'
    has_system_prompt_col = "SYSTEM_PROMPT" in df.columns
    default_system_prompt = None

    if has_system_prompt_col:
        logging.info("Using 'SYSTEM_PROMPT' column from the dataframe.")
        print("Using 'SYSTEM_PROMPT' column from the dataframe.")
    else:
        logging.info("'SYSTEM_PROMPT' column not found in dataframe. Looking for SYSTEM_PROMPT.md file in the current working directory...")
        print("'SYSTEM_PROMPT' column not found in dataframe. Looking for SYSTEM_PROMPT.md file in the current working directory...")
        if os.path.exists("./SYSTEM_PROMPT.md"):
            try:
                with open("./SYSTEM_PROMPT.md", encoding="utf-8") as f:
                    default_system_prompt = f.read()
                logging.info("Successfully loaded system prompt from './SYSTEM_PROMPT.md'")
                print("Successfully loaded system prompt from './SYSTEM_PROMPT.md'")
            except Exception as e:
                logging.error(f"Failed to read './SYSTEM_PROMPT.md': {str(e)}")
                return
        else:
            logging.error("SYSTEM_PROMPT.md file not found in current directory.")
            print("Error: SYSTEM_PROMPT.md file not found in current directory.")
            return

    # Initialize managers
    rate_limiter = SlidingWindowRateLimiter(
        max_requests=REQUESTS_PER_MINUTE, time_window=60
    )
    results_manager = ResultsManager(DIR_RESULT, PKL_FILE_NAME)

    # Prepare list of all prompts with appropriate system prompt
    all_prompts = []
    for _, row in df.iterrows():
        prompt_id = row[PROMPT_INDEX_COLUMN]
        prompt_text = row[PROMPT_COLUMN]
        
        if has_system_prompt_col:
            # Safely handle empty/nan system prompts
            row_sys_prompt = row["SYSTEM_PROMPT"]
            system_prompt = row_sys_prompt if pd.notna(row_sys_prompt) else ""
        else:
            system_prompt = default_system_prompt

        all_prompts.append({
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "system_prompt": system_prompt
        })

    total_prompts = len(all_prompts)
    logging.info(f"Total prompts to process: {total_prompts}")
    print(f"Total prompts to process: {total_prompts}")

    # Filter out already processed prompts
    pending_prompts = [
        p for p in all_prompts if not results_manager.is_processed(p["prompt_id"])
    ]

    if len(pending_prompts) < total_prompts:
        already_processed = total_prompts - len(pending_prompts)
        logging.info(f"Skipping {already_processed} already processed prompts")
        print(f"Skipping {already_processed} already processed prompts")

    # Process in passes with retry logic
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        if not pending_prompts:
            logging.info("All prompts processed successfully!")
            print("All prompts processed successfully!")
            break

        logging.info(f"\n{'='*60}")
        logging.info(
            f"Pass {attempt}/{MAX_RETRY_ATTEMPTS}: Processing {len(pending_prompts)} prompts"
        )
        logging.info(f"{'='*60}")
        print(
            f"\nPass {attempt}/{MAX_RETRY_ATTEMPTS}: Processing {len(pending_prompts)} prompts"
        )

        failed_prompts = []
        pass_start_time = datetime.now()

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all prompts for this pass
            future_to_prompt = {
                executor.submit(
                    process_single_prompt,
                    client,
                    prompt_data,
                    rate_limiter,
                    results_manager,
                    df,
                ): prompt_data
                for prompt_data in pending_prompts
            }

            # Process completed requests
            completed_count = 0
            for future in as_completed(future_to_prompt):
                prompt_data = future_to_prompt[future]
                completed_count += 1

                try:
                    prompt_id, success, error_msg = future.result()

                    if not success:
                        failed_prompts.append(prompt_data)
                        logging.warning(
                            f"[{completed_count}/{len(pending_prompts)}] "
                            f"Failed prompt ID {prompt_id}: {error_msg}"
                        )
                    else:
                        logging.info(
                            f"[{completed_count}/{len(pending_prompts)}] "
                            f"Completed prompt ID {prompt_id}"
                        )

                except Exception as e:
                    failed_prompts.append(prompt_data)
                    logging.error(
                        f"[{completed_count}/{len(pending_prompts)}] "
                        f"Exception processing prompt ID {prompt_data['prompt_id']}: {str(e)}"
                    )

        pass_end_time = datetime.now()
        pass_duration = (pass_end_time - pass_start_time).total_seconds()

        # Pass summary
        success_count = len(pending_prompts) - len(failed_prompts)
        logging.info(f"\nPass {attempt} completed in {pass_duration:.2f} seconds")
        logging.info(f"Successful: {success_count}, Failed: {len(failed_prompts)}")
        print(
            f"\nPass {attempt} summary: {success_count} successful, {len(failed_prompts)} failed"
        )

        # Update pending prompts for next pass
        pending_prompts = failed_prompts

        # If this was the last attempt, log final failures
        if pending_prompts and attempt == MAX_RETRY_ATTEMPTS:
            logging.warning(
                f"\nReached maximum retry attempts. {len(pending_prompts)} prompts still failed:"
            )
            for p in pending_prompts:
                logging.warning(f"  - Prompt ID: {p['prompt_id']}")

    # Final summary
    total_successful = results_manager.get_results_count()
    total_failed = total_prompts - total_successful

    logging.info(f"\n{'='*60}")
    logging.info(f"FINAL SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Total prompts: {total_prompts}")
    logging.info(f"Successfully processed: {total_successful}")
    logging.info(f"Failed: {total_failed}")
    logging.info(f"Results saved to: {DIR_RESULT}")

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {total_successful}/{total_prompts}")
    print(f"Results saved to: {DIR_RESULT}")
    print(f"{'='*60}")


def main():
    try:
        path_to_prompts = input(
            "Enter the path to the PKL file containing prompts: "
        ).strip()

        if not os.path.exists(path_to_prompts):
            print(f"Error: File '{path_to_prompts}' not found.")
            sys.exit(1)

        client = setup_openai_api()
        process_prompts(client, path_to_prompts)

    except KeyboardInterrupt:
        logging.info("\nProcess interrupted by user. Results saved up to this point.")
        print("\nProcess interrupted by user. Results saved up to this point.")
        sys.exit(0)
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {str(e)}")
        print(
            f"An unexpected error occurred. Please check the log file in {DIR_RESULT} for details."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
