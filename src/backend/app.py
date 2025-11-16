import asyncio
import json
import os
import time
import uuid
import csv
from io import StringIO
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv

from auth import require_auth

# Load environment variables
load_dotenv()
# Create FastAPI app instance
app = FastAPI()

# Add auth middleware FIRST (before CORS)
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Skip auth for CORS preflight requests
    if request.method == "OPTIONS":
        return await call_next(request)
    await require_auth(request)
    response = await call_next(request)
    return response

# Add CORS middleware AFTER auth middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load scenario data from JSON
data_path = Path(__file__).parent / "data" / "scenarios.json"
scenarios_data = {}
if data_path.exists():
    with open(data_path, "r", encoding="utf-8") as f:
        scenarios_data = json.load(f)

# Load pricing data
pricing_path = Path(__file__).parent / "data" / "pricing.json"
pricing_data = {}
if pricing_path.exists():
    with open(pricing_path, "r", encoding="utf-8") as f:
        pricing_data = json.load(f)

# Load ground truth data
ground_truth_path = Path(__file__).parent / "data" / "ground_truth.json"
ground_truth_data = {}
if ground_truth_path.exists():
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)


class PromptRequest(BaseModel):
    prompt: str
    ground_truth: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    model_type: str = "router"  # "router" or "benchmark"


class DatasetEvaluationJob(BaseModel):
    """Represents a dataset evaluation job"""
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: int  # 0-100
    total_rows: int
    processed_rows: int
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None


# In-memory job storage (in production, use a database)
evaluation_jobs: Dict[str, DatasetEvaluationJob] = {}

# Flat router classification surcharge (USD per 1M input tokens)
ROUTER_CLASSIFICATION_INPUT_RATE = 0.14


def augment_prompt_with_context(prompt_text: str) -> str:
    """Check if the prompt matches a scenario and augment with RAG data if available"""
    for dept_scenarios in scenarios_data.values():
        for scenario in dept_scenarios:
            if scenario["prompt"] == prompt_text:
                if scenario.get("source_data_file"):
                    try:
                        context_file_path = Path(__file__).parent / "data" / "scenario_source_data" / scenario["source_data_file"]
                        if context_file_path.exists():
                            with open(context_file_path, "r", encoding="utf-8") as f:
                                context_data = f.read()
                            return f"{prompt_text}\n\nContext Data:\n{context_data}"
                    except Exception as e:
                        print(f"Error loading context data: {e}")
                break
    return prompt_text


def get_scenario_id_from_prompt(prompt_text: str) -> Optional[str]:
    """Find the scenario ID that matches the given prompt"""
    for dept_scenarios in scenarios_data.values():
        for scenario in dept_scenarios:
            if scenario["prompt"] == prompt_text:
                return scenario["id"]
    return None


async def evaluate_responses_comparatively(
    scenario_id: str,
    router_response: str,
    router_model_name: str,
    benchmark_response: str,
    benchmark_model_name: str,
    custom_ground_truth: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use LLM-as-a-judge to evaluate both responses together in a single comparative evaluation.
    This ensures scores are relative to each other and the ground truth.
    
    Args:
        scenario_id: The scenario ID (e.g., "fin-1")
        router_response: The router model's response
        router_model_name: Name of the router model
        benchmark_response: The benchmark model's response
        benchmark_model_name: Name of the benchmark model
        custom_ground_truth: Optional custom ground truth provided by the user
    
    Returns:
        Dictionary containing evaluations for both models with relative scores
    """
    # Use custom ground truth if provided, otherwise look it up from data
    if custom_ground_truth:
        expected_answer = custom_ground_truth
        evaluation_criteria = "Evaluate the responses based on their alignment with the provided ground truth answer and overall quality."
    elif scenario_id in ground_truth_data:
        ground_truth = ground_truth_data[scenario_id]
        expected_answer = ground_truth["expected_answer"]
        evaluation_criteria = ground_truth["evaluation_criteria"]
    else:
        return {
            "router": {
                "score": None,
                "reasoning": "No ground truth available for this scenario",
                "error": "Ground truth not found"
            },
            "benchmark": {
                "score": None,
                "reasoning": "No ground truth available for this scenario",
                "error": "Ground truth not found"
            }
        }
    
    # Create comparative evaluation prompt for the judge model
    judge_prompt = f"""You are an expert evaluator assessing the accuracy and quality of AI model responses. You will compare TWO different model responses against the same ground truth simultaneously to provide relative, consistent scoring.

**Expected Answer (Ground Truth):**
{expected_answer}

**Evaluation Criteria:**
{evaluation_criteria}

**Model A ({router_model_name}) Response:**
{router_response}

**Model B ({benchmark_model_name}) Response:**
{benchmark_response}

**Instructions:**
1. Compare BOTH responses against the expected answer and evaluation criteria
2. Consider the responses relative to EACH OTHER as well as to the ground truth
3. Assign accuracy scores from 0-100 for EACH model:
   - 90-100: Excellent - Meets or exceeds all criteria, highly accurate
   - 75-89: Good - Meets most criteria with minor gaps
   - 60-74: Adequate - Meets some criteria but has notable gaps
   - 40-59: Poor - Missing significant elements or has accuracy issues
   - 0-39: Failing - Largely incorrect or missing most required elements

4. Both models CAN receive the same score if they perform equally well
5. Ensure scoring is relative - if one is clearly better, the scores should reflect that difference
6. Provide detailed reasoning for each score, citing specific strengths and weaknesses
7. Note any significant differences in quality between the two responses

**Output Format (respond ONLY with valid JSON):**
{{
    "model_a": {{
        "accuracy_score": <number 0-100>,
        "reasoning": "<detailed explanation of score>",
        "strengths": ["<strength 1>", "<strength 2>"],
        "weaknesses": ["<weakness 1>", "<weakness 2>"],
        "key_gaps": ["<gap 1>", "<gap 2>"]
    }},
    "model_b": {{
        "accuracy_score": <number 0-100>,
        "reasoning": "<detailed explanation of score>",
        "strengths": ["<strength 1>", "<strength 2>"],
        "weaknesses": ["<weakness 1>", "<weakness 2>"],
        "key_gaps": ["<gap 1>", "<gap 2>"]
    }},
    "comparative_analysis": "<brief comparison highlighting key differences between the two responses>"
}}"""
    
    try:
        # Use benchmark model as the judge
        result = await call_azure_openai(judge_prompt, is_benchmark=True)
        
        # Parse the JSON response
        evaluation_text = result["output"].strip()
        
        # Extract JSON from response (handles cases where model adds extra text)
        json_start = evaluation_text.find('{')
        json_end = evaluation_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = evaluation_text[json_start:json_end]
            evaluation = json.loads(json_str)
            
            return {
                "router": {
                    "score": evaluation.get("model_a", {}).get("accuracy_score", 0),
                    "reasoning": evaluation.get("model_a", {}).get("reasoning", ""),
                    "strengths": evaluation.get("model_a", {}).get("strengths", []),
                    "weaknesses": evaluation.get("model_a", {}).get("weaknesses", []),
                    "key_gaps": evaluation.get("model_a", {}).get("key_gaps", []),
                    "model_evaluated": router_model_name
                },
                "benchmark": {
                    "score": evaluation.get("model_b", {}).get("accuracy_score", 0),
                    "reasoning": evaluation.get("model_b", {}).get("reasoning", ""),
                    "strengths": evaluation.get("model_b", {}).get("strengths", []),
                    "weaknesses": evaluation.get("model_b", {}).get("weaknesses", []),
                    "key_gaps": evaluation.get("model_b", {}).get("key_gaps", []),
                    "model_evaluated": benchmark_model_name
                },
                "comparative_analysis": evaluation.get("comparative_analysis", "")
            }
        else:
            # Fallback: could not parse
            return {
                "router": {
                    "score": None,
                    "reasoning": evaluation_text,
                    "error": "Could not parse structured evaluation",
                    "model_evaluated": router_model_name
                },
                "benchmark": {
                    "score": None,
                    "reasoning": evaluation_text,
                    "error": "Could not parse structured evaluation",
                    "model_evaluated": benchmark_model_name
                }
            }
            
    except Exception as e:
        print(f"Error in comparative accuracy evaluation: {e}")
        return {
            "router": {
                "score": None,
                "reasoning": f"Evaluation failed: {str(e)}",
                "error": str(e),
                "model_evaluated": router_model_name
            },
            "benchmark": {
                "score": None,
                "reasoning": f"Evaluation failed: {str(e)}",
                "error": str(e),
                "model_evaluated": benchmark_model_name
            }
        }





async def call_azure_openai(prompt: str, is_benchmark: bool = False) -> Dict[str, Any]:
    """Call Azure OpenAI API without blocking the event loop."""
    endpoint_key = "AZURE_OPENAI_BENCHMARK_ENDPOINT" if is_benchmark else "AZURE_OPENAI_ENDPOINT"
    api_key_key = "AZURE_OPENAI_BENCHMARK_API_KEY" if is_benchmark else "AZURE_OPENAI_API_KEY"
    deployment_key = "AZURE_OPENAI_BENCHMARK_DEPLOYMENT_NAME" if is_benchmark else "AZURE_OPENAI_DEPLOYMENT_NAME"

    endpoint = os.getenv(endpoint_key)
    api_key = os.getenv(api_key_key)
    deployment_name = os.getenv(deployment_key)

    if not all([endpoint, api_key, deployment_name]):
        raise HTTPException(
            status_code=500,
            detail=f"Azure OpenAI credentials not configured for {'benchmark' if is_benchmark else 'router'} model"
        )

    def _invoke_sync_client() -> Dict[str, Any]:
        try:
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            )

            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )

            return {
                "model_type": response.model,
                "output": response.choices[0].message.content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        except Exception as exc:  # pragma: no cover - surfaced through HTTPException
            print(f"Azure OpenAI API error: {exc}")
            print(f"Error type: {type(exc).__name__}")
            print(f"Endpoint: {endpoint}")
            print(f"Deployment: {deployment_name}")
            import traceback
            traceback.print_exc()
            raise exc

    try:
        return await asyncio.to_thread(_invoke_sync_client)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Azure OpenAI API error for {'benchmark' if is_benchmark else 'router'} model: {str(e)}"
        )


async def stream_azure_openai_chat(prompt: str, is_benchmark: bool = False):
    """
    Stream Azure OpenAI chat completion responses.
    Yields Server-Sent Events (SSE) formatted chunks.
    """
    endpoint_key = "AZURE_OPENAI_BENCHMARK_ENDPOINT" if is_benchmark else "AZURE_OPENAI_ENDPOINT"
    api_key_key = "AZURE_OPENAI_BENCHMARK_API_KEY" if is_benchmark else "AZURE_OPENAI_API_KEY"
    deployment_key = "AZURE_OPENAI_BENCHMARK_DEPLOYMENT_NAME" if is_benchmark else "AZURE_OPENAI_DEPLOYMENT_NAME"

    endpoint = os.getenv(endpoint_key)
    api_key = os.getenv(api_key_key)
    deployment_name = os.getenv(deployment_key)

    if not all([endpoint, api_key, deployment_name]):
        error_msg = f"Azure OpenAI credentials not configured for {'benchmark' if is_benchmark else 'router'} model"
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
        return

    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
        )

        # Create the streaming request
        def get_stream():
            return client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )
        
        # Run the sync call in thread pool
        stream = await asyncio.to_thread(get_stream)
        
        # Process the stream
        model_name = None
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if getattr(chunk, "model", None) and not model_name:
                    model_name = chunk.model
                if delta.content:
                    content = delta.content
                    yield f"data: {json.dumps({'content': content})}\n\n"
        
        # Send completion signal
        if not model_name:
            model_name = deployment_name
        yield f"data: {json.dumps({'done': True, 'model': model_name})}\n\n"
        
    except Exception as exc:
        error_msg = f"Azure OpenAI streaming error: {str(exc)}"
        print(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"


@app.get("/")
async def hello_world():
    await asyncio.sleep(1)
    return {"message": "Hello World"}


@app.get("/api/scenarios/{department}")
async def get_scenarios(department: str):
    """Get scenarios for a specific department"""
    return scenarios_data.get(department, [])


@app.get("/api/pricing")
async def get_pricing():
    """Get pricing configuration"""
    return pricing_data


@app.get("/api/ground-truth/{scenario_id}")
async def get_ground_truth(scenario_id: str):
    """Get ground truth data for a specific scenario"""
    if scenario_id not in ground_truth_data:
        raise HTTPException(status_code=404, detail="Ground truth not found for this scenario")
    return ground_truth_data[scenario_id]


@app.post("/api/route")
async def route_prompt(request: PromptRequest):
    """Route a prompt through the model router"""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Augment prompt with RAG context data
    augmented_prompt = augment_prompt_with_context(request.prompt)
    
    # Measure processing time
    start_time = time.perf_counter()
    result = await call_azure_openai(augmented_prompt)
    end_time = time.perf_counter()
    
    result["server_processing_ms"] = int((end_time - start_time) * 1000)
    return result


@app.post("/api/benchmark")
async def benchmark_only(request: PromptRequest):
    """Get benchmark response only"""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Augment prompt with RAG context data
    augmented_prompt = augment_prompt_with_context(request.prompt)
    
    # Measure processing time
    start_time = time.perf_counter()
    result = await call_azure_openai(augmented_prompt, is_benchmark=True)
    end_time = time.perf_counter()
    
    result["server_processing_ms"] = int((end_time - start_time) * 1000)
    return result


@app.post("/api/route-comparison")
async def route_comparison(request: PromptRequest):
    """Get both router and benchmark responses for comparison"""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Augment prompt with RAG context data
    augmented_prompt = augment_prompt_with_context(request.prompt)
    
    # Get both responses in parallel
    router_task = call_azure_openai(augmented_prompt, is_benchmark=False)
    benchmark_task = call_azure_openai(augmented_prompt, is_benchmark=True)
    
    router_result, benchmark_result = await asyncio.gather(router_task, benchmark_task)
    
    return {
        "router": router_result,
        "benchmark": benchmark_result
    }


@app.post("/api/accuracy-comparison")
async def accuracy_comparison(request: PromptRequest):
    """
    Get router and benchmark responses, then evaluate both against ground truth.
    Returns responses with accuracy evaluations.
    
    Supports both pre-built scenarios (with stored ground truth) and custom scenarios
    (with user-provided ground truth).
    """
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Find scenario ID for this prompt (for pre-built scenarios)
    scenario_id = get_scenario_id_from_prompt(request.prompt)
    
    # If no pre-built scenario found and no custom ground truth provided, return error
    if not scenario_id and not request.ground_truth:
        raise HTTPException(
            status_code=400, 
            detail="Prompt does not match any known scenario. Please provide custom ground truth."
        )
    
    # Use a placeholder ID for custom scenarios if needed
    if not scenario_id:
        scenario_id = "custom"
    
    # Augment prompt with RAG context data (only for known scenarios)
    if get_scenario_id_from_prompt(request.prompt):
        augmented_prompt = augment_prompt_with_context(request.prompt)
    else:
        augmented_prompt = request.prompt
    
    # Get both responses in parallel
    start_time = time.perf_counter()
    router_task = call_azure_openai(augmented_prompt, is_benchmark=False)
    benchmark_task = call_azure_openai(augmented_prompt, is_benchmark=True)
    
    router_result, benchmark_result = await asyncio.gather(router_task, benchmark_task)
    response_time = time.perf_counter() - start_time
    
    # Evaluate both responses together in a single comparative evaluation
    eval_start = time.perf_counter()
    comparative_evaluation = await evaluate_responses_comparatively(
        scenario_id,
        router_result["output"],
        router_result["model_type"],
        benchmark_result["output"],
        benchmark_result["model_type"],
        request.ground_truth
    )
    evaluation_time = time.perf_counter() - eval_start
    
    return {
        "scenario_id": scenario_id,
        "router": {
            **router_result,
            "accuracy_evaluation": comparative_evaluation["router"]
        },
        "benchmark": {
            **benchmark_result,
            "accuracy_evaluation": comparative_evaluation["benchmark"]
        },
        "comparative_analysis": comparative_evaluation.get("comparative_analysis", ""),
        "timing": {
            "response_generation_ms": int(response_time * 1000),
            "accuracy_evaluation_ms": int(evaluation_time * 1000),
            "total_ms": int((response_time + evaluation_time) * 1000)
        }
    }


async def process_dataset_evaluation(job_id: str, csv_rows: List[Dict[str, str]]):
    """
    Background task to process dataset evaluation.
    Runs router and benchmark for each row, evaluates accuracy if ground truth provided.
    """
    try:
        job = evaluation_jobs[job_id]
        job.status = "processing"
        job.total_rows = len(csv_rows)
        job.processed_rows = 0
        job.results = []
        
        for index, row in enumerate(csv_rows):
            try:
                prompt = row.get("prompt", "").strip()
                ground_truth = row.get("ground_truth", "").strip() or None
                
                if not prompt:
                    # Skip empty prompts
                    job.processed_rows += 1
                    job.progress = int((job.processed_rows / job.total_rows) * 100)
                    continue
                
                # Run router and benchmark sequentially to get accurate individual latencies
                # Router call
                router_start = time.perf_counter()
                router_result = await call_azure_openai(prompt, is_benchmark=False)
                router_latency_ms = int((time.perf_counter() - router_start) * 1000)
                
                # Benchmark call
                benchmark_start = time.perf_counter()
                benchmark_result = await call_azure_openai(prompt, is_benchmark=True)
                benchmark_latency_ms = int((time.perf_counter() - benchmark_start) * 1000)
                
                # Calculate costs
                router_cost, router_pricing_warning, router_cost_breakdown = calculate_cost_for_result(
                    router_result,
                    include_router_surcharge=True
                )
                benchmark_cost, benchmark_pricing_warning, benchmark_cost_breakdown = calculate_cost_for_result(
                    benchmark_result
                )
                
                # Prepare result for this row
                row_result = {
                    "row_index": index,
                    "prompt": prompt,
                    "router": {
                        "model_type": router_result["model_type"],
                        "output": router_result["output"],
                        "latency_ms": router_latency_ms,
                        "prompt_tokens": router_result["prompt_tokens"],
                        "completion_tokens": router_result["completion_tokens"],
                        "cost": router_cost
                    },
                    "benchmark": {
                        "model_type": benchmark_result["model_type"],
                        "output": benchmark_result["output"],
                        "latency_ms": benchmark_latency_ms,
                        "prompt_tokens": benchmark_result["prompt_tokens"],
                        "completion_tokens": benchmark_result["completion_tokens"],
                        "cost": benchmark_cost
                    }
                }

                if router_pricing_warning:
                    row_result["router"]["pricing_warning"] = router_pricing_warning
                if router_cost_breakdown:
                    row_result["router"]["cost_breakdown"] = router_cost_breakdown
                if benchmark_pricing_warning:
                    row_result["benchmark"]["pricing_warning"] = benchmark_pricing_warning
                if benchmark_cost_breakdown:
                    row_result["benchmark"]["cost_breakdown"] = benchmark_cost_breakdown
                
                # If ground truth provided, evaluate accuracy using comparative evaluation
                if ground_truth:
                    eval_start = time.perf_counter()
                    comparative_evaluation = await evaluate_responses_comparatively(
                        "custom",
                        router_result["output"],
                        router_result["model_type"],
                        benchmark_result["output"],
                        benchmark_result["model_type"],
                        ground_truth
                    )
                    
                    row_result["router"]["accuracy"] = comparative_evaluation["router"].get("score")
                    row_result["benchmark"]["accuracy"] = comparative_evaluation["benchmark"].get("score")
                    row_result["accuracy_evaluation_time_ms"] = int((time.perf_counter() - eval_start) * 1000)
                
                job.results.append(row_result)
                job.processed_rows += 1
                job.progress = int((job.processed_rows / job.total_rows) * 100)
                
            except Exception as row_error:
                print(f"Error processing row {index}: {row_error}")
                # Continue with next row even if one fails
                job.processed_rows += 1
                job.progress = int((job.processed_rows / job.total_rows) * 100)
                continue
        
        # Job completed successfully
        job.status = "completed"
        job.progress = 100
        job.completed_at = datetime.utcnow().isoformat()
        
    except Exception as e:
        print(f"Dataset evaluation job {job_id} failed: {e}")
        job = evaluation_jobs[job_id]
        job.status = "failed"
        job.error_message = str(e)
        job.completed_at = datetime.utcnow().isoformat()


def calculate_cost_for_result(
    result: Dict[str, Any],
    include_router_surcharge: bool = False
) -> Tuple[Optional[float], Optional[str], Optional[Dict[str, float]]]:
    """Calculate cost for a model result using pricing data.

    Returns:
        Tuple of (total_cost, warning message, cost_breakdown). total_cost may be None if
        pricing cannot be determined, and cost_breakdown will be None in that scenario.
    """
    model_type = result.get("model_type", "default")
    prompt_tokens = result.get("prompt_tokens", 0)
    completion_tokens = result.get("completion_tokens", 0)
    
    model_pricing = pricing_data.get("models", {}).get(model_type)
    warning: Optional[str] = None

    if not model_pricing:
        default_pricing = pricing_data.get("models", {}).get("default")
        if default_pricing:
            warning = (
                f"Pricing not configured for model '{model_type}'. Using 'default' fallback rates. "
                "This usually means the router selected a newer model that this demo hasn't mapped yet."
            )
            model_pricing = default_pricing
        else:
            warning = (
                f"Pricing not configured for model '{model_type}' and no fallback rates are defined."
            )
            return None, warning, None
    
    input_cost = (prompt_tokens / 1000000) * model_pricing["input_per_1m"]
    output_cost = (completion_tokens / 1000000) * model_pricing["output_per_1m"]
    router_surcharge = 0.0

    if include_router_surcharge:
        router_surcharge = (prompt_tokens / 1000000) * ROUTER_CLASSIFICATION_INPUT_RATE
    
    total_cost = input_cost + output_cost + router_surcharge
    breakdown = {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "router_surcharge": router_surcharge
    }

    return round(total_cost, 8), warning, breakdown


@app.post("/api/dataset-evaluation/submit")
async def submit_dataset_evaluation(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Submit a CSV file for dataset evaluation.
    Returns a job_id for tracking progress.
    
    Expected CSV format:
    prompt,ground_truth (optional)
    "Your prompt here","Expected answer here"
    "Another prompt","Another expected answer"
    
    Note: This endpoint returns immediately. The actual processing happens in the background.
    Poll /api/dataset-evaluation/status/{job_id} to track progress.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read and parse CSV
        contents = await file.read()
        csv_string = contents.decode('utf-8')
        csv_reader = csv.DictReader(StringIO(csv_string))
        
        rows = []
        for row in csv_reader:
            rows.append(row)
        
        if len(rows) == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Limit dataset size to 12 rows
        if len(rows) > 12:
            raise HTTPException(
                status_code=400, 
                detail=f"Dataset too large. Maximum 12 rows allowed. Your file has {len(rows)} rows."
            )
        
        # Validate CSV has required columns
        if 'prompt' not in rows[0]:
            raise HTTPException(
                status_code=400, 
                detail="CSV must contain a 'prompt' column. Optional: 'ground_truth' column"
            )
        
        # Create job
        job_id = str(uuid.uuid4())
        job = DatasetEvaluationJob(
            job_id=job_id,
            status="queued",
            progress=0,
            total_rows=len(rows),
            processed_rows=0,
            created_at=datetime.utcnow().isoformat()
        )
        
        evaluation_jobs[job_id] = job
        
        # Add background task using FastAPI's BackgroundTasks (proper async handling)
        background_tasks.add_task(process_dataset_evaluation, job_id, rows)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "total_rows": len(rows),
            "message": "Dataset evaluation started"
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid CSV file encoding. Please use UTF-8")
    except csv.Error as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    except Exception as e:
        print(f"Error submitting dataset evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")


@app.get("/api/dataset-evaluation/status/{job_id}")
async def get_evaluation_status(job_id: str):
    """Get the status of a dataset evaluation job"""
    if job_id not in evaluation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = evaluation_jobs[job_id]
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "total_rows": job.total_rows,
        "processed_rows": job.processed_rows,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "error_message": job.error_message
    }


@app.get("/api/dataset-evaluation/results/{job_id}")
async def get_evaluation_results(job_id: str):
    """Get the results of a completed dataset evaluation job"""
    if job_id not in evaluation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = evaluation_jobs[job_id]
    
    if job.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed yet. Current status: {job.status}"
        )
    
    # Calculate summary statistics
    if job.results:
        total_router_latency = sum(r["router"]["latency_ms"] for r in job.results)
        total_benchmark_latency = sum(r["benchmark"]["latency_ms"] for r in job.results)
        router_costs = [r["router"].get("cost") for r in job.results if r["router"].get("cost") is not None]
        benchmark_costs = [r["benchmark"].get("cost") for r in job.results if r["benchmark"].get("cost") is not None]
        total_router_cost = sum(router_costs) if router_costs else 0
        total_benchmark_cost = sum(benchmark_costs) if benchmark_costs else 0
        
        # Calculate accuracy averages if available
        router_accuracies = [r["router"].get("accuracy") for r in job.results if r["router"].get("accuracy") is not None]
        benchmark_accuracies = [r["benchmark"].get("accuracy") for r in job.results if r["benchmark"].get("accuracy") is not None]
        
        summary = {
            "total_rows": len(job.results),
            "avg_router_latency_ms": int(total_router_latency / len(job.results)),
            "avg_benchmark_latency_ms": int(total_benchmark_latency / len(job.results)),
            "total_router_cost": round(total_router_cost, 8) if router_costs else 0,
            "total_benchmark_cost": round(total_benchmark_cost, 8) if benchmark_costs else 0,
            "cost_savings_percent": round(((total_benchmark_cost - total_router_cost) / total_benchmark_cost * 100), 2) if benchmark_costs and total_benchmark_cost > 0 else 0,
            "latency_improvement_percent": round(((total_benchmark_latency - total_router_latency) / total_benchmark_latency * 100), 2) if total_benchmark_latency > 0 else 0
        }
        
        if router_accuracies:
            summary["avg_router_accuracy"] = round(sum(router_accuracies) / len(router_accuracies), 2)
        if benchmark_accuracies:
            summary["avg_benchmark_accuracy"] = round(sum(benchmark_accuracies) / len(benchmark_accuracies), 2)
    else:
        summary = {}
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "completed_at": job.completed_at,
        "summary": summary,
        "results": job.results
    }


@app.delete("/api/dataset-evaluation/job/{job_id}")
async def delete_evaluation_job(job_id: str):
    """Delete a dataset evaluation job"""
    if job_id not in evaluation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del evaluation_jobs[job_id]
    return {"message": "Job deleted successfully"}


@app.post("/api/chat/zava")
async def chat_zava(request: ChatRequest):
    """
    Chat endpoint for Zava assistant with streaming support.
    Returns a Server-Sent Events (SSE) stream of the response.
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    # Zava company context
    zava_context = """You are a concise and helpful AI assistant for Zava, a cutting-edge smart sportswear company.

RESPONSE STYLE:
- Keep responses brief and focused (2-3 sentences for simple questions, max 1 short paragraph for complex ones).
- Get to the point quickly while remaining friendly and engaging.

COMPANY OVERVIEW:
- Mission: Make elite-level athletic performance insights accessible to all athletes.
- Location: Headquarters in Palo Alto, CA (1 Hacker Way, Palo Alto, CA 94301).
- Founded: 2019 with vision to revolutionize athletic performance through smart technology.
- Contact: info@zava.com, hello@zava.com, support@zava.com | Phone: +1 (555) 123-ZAVA

PRODUCTS:
Smart Jerseys:
1. Zava Pro Jersey ($299) - Professional-grade smart jersey with 12 integrated biometric sensors, heart rate monitoring, temperature regulation, motion tracking, sweat analysis. 72-hour battery life, Bluetooth 5.0 + WiFi connectivity, machine washable.
2. Zava Training Jersey ($199) - Training-focused jersey with 6 key monitoring points, basic monitoring, comfort fit, quick dry, team sync. 48-hour battery life, Bluetooth 5.0, machine washable.

Smart Cleats:
1. Zava Elite Cleats ($399) - Revolutionary smart cleats with 16 pressure points, pressure mapping, gait analysis, speed tracking, balance optimization. 48-hour battery life, Bluetooth 5.0, 280g per cleat.
2. Zava Speed Cleats ($299) - Lightweight cleats (240g per cleat) with 8 motion sensors, sprint analysis, acceleration tracking, direction changes. 36-hour battery life, Bluetooth 5.0.

TECHNOLOGY:
- Biometric Monitoring: Continuous heart rate, breathing rate, stress level tracking.
- Temperature Control: Smart fabric regulates body temperature and moisture.
- Motion Analysis: Advanced accelerometers track movement patterns and posture.
- Pressure Mapping: Detailed foot pressure analysis to optimize stride.
- Gait Analysis: Comprehensive running form analysis with improvement suggestions.
- Balance Optimization: Center of gravity tracking for enhanced stability.
- Mobile App: iOS & Android with comprehensive analytics dashboard.
- Connectivity: Bluetooth 5.0 + WiFi for seamless data transfer.

FEATURED ATHLETES:
- Marcus Johnson (Football): "Zava's smart jersey helped me optimize my training." 23% performance increase.
- Sofia Rodriguez (Soccer): "The smart cleats revolutionized my running patterns." 31% sprint speed improvement.
- James Chen (Basketball): "Training with Zava feels like having a personal coach." 18% jump height improvement.
- Emma Thompson (Track & Field): "Zava's analytics helped me break my personal record." 12% race time improvement.

LEADERSHIP:
- Dr. Sarah Chen (CEO): Former Olympic athlete, PhD in Sports Science from Stanford.
- Marcus Rodriguez (CTO): Ex-Apple engineer with 15 years in wearable technology.
- Dr. Elena Volkova (Head of Research): Leading sports biomechanics researcher.
- James Park (VP of Product): Former Nike product manager.

CUSTOMER PROFILE: Emily Thompson (ID: C1024)
- Age: 35 | Member: 24 months | Total Spend: $4,800 | Avg Monthly: $200
- Preferred Categories: Jerseys, Cleats, Accessories

PURCHASE HISTORY (Last 6 Months):
| Order | Date | Items | Amount | Discount | Delivered | Returned |
|-------|------|-------|--------|----------|-----------|----------|
| O5678 | 2023-03-15 | Zava Training Jersey (Summer Edition), Sun Cap | $150 | 10% | 2023-03-19 | No |
| O5721 | 2023-04-10 | Zava Speed Cleats | $120 | 15% | 2023-04-13 | No |
| O5789 | 2023-05-05 | Cooling Neck Wrap | $80 | 0% | 2023-05-25 | Yes |
| O5832 | 2023-06-18 | Zava Training Cleats | $90 | 5% | 2023-06-21 | No |
| O5890 | 2023-07-22 | Zava Elite Cleats (pair) | $300 | 20% | 2023-08-05 | No |
| O5935 | 2023-08-30 | Performance Training Jacket | $110 | 0% | 2023-09-03 | Yes |
| O5970 | 2023-09-12 | Compression Leggings, Sports Bra | $130 | 25% | 2023-09-18 | No |

CUSTOMER SERVICE INTERACTIONS:
Interaction 1 (Live Chat, 2023-06-20):
- Emily wanted to swap order O5789 for a different color.
- Agent initiated return process successfully.

Interaction 2 (Phone Call, 2023-07-25):
- Emily called about order O5890 (Elite Cleats) to ensure timely delivery for an important soccer game.
- Agent confirmed delivery was on track.

LOYALTY PROGRAM:
- Total Points Earned: 4,800
- Points Redeemed: 3,600
- Current Balance: 1,200 points
- Points Expiring Next Month: 1,200

RECENT WEBSITE ACTIVITY:
| Date | Pages Visited | Time (Minutes) |
|------|---------------|----------------|
| 2023-09-10 | Homepage, New Arrivals, Jerseys | 15 |
| 2023-09-11 | Account Settings, Subscription Details | 5 |
| 2023-09-12 | FAQ, Return Policy | 3 |
| 2023-09-14 | Sale Items, Accessories | 10 |

Please provide helpful, accurate, and engaging responses about Zava. Use the detailed information above to answer user questions comprehensively."""

    user_message = request.message.strip()

    # Combine context with user message
    full_prompt = f"{zava_context}\n\nUser Question: {user_message}\n\nAssistant Response:"
    
    # Log the size of the prompt
    print(f"Zava chat prompt length: {len(full_prompt)} characters")

    # Determine which model to use
    use_benchmark = request.model_type.lower() == "benchmark"
    
    # Return streaming response
    return StreamingResponse(
        stream_azure_openai_chat(full_prompt, is_benchmark=use_benchmark),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        }
    )
