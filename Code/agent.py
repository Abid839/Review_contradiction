import os
import json
import sys
import re
import asyncio
import nest_asyncio
import logging
import time

# Enable nested event loops (useful in Jupyter)
nest_asyncio.apply()

# Setup logging to a file for debugging
logging.basicConfig(filename='agent_debug.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Replace with your actual OpenAI API key
API_KEY = API_KEY
# Import AutoGen components (ensure you have installed autogen-agentchat and autogen-ext[openai])
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ImportError as e:
    logging.error("AutoGen framework is not installed. Please install autogen-agentchat and autogen-ext[openai].")
    raise

def extract_reviews(data):
    """
    Recursively traverse the JSON data and extract all string values from keys that
    match the pattern "Review_<number>_full" (case-insensitive) or the key "app_pairing_content".
    """
    reviews = []
    review_pattern = re.compile(r'^review_\d+_full$', re.IGNORECASE)
    
    if isinstance(data, dict):
        for key, value in data.items():
            if review_pattern.match(key) or key.lower() == "app_pairing_content":
                if isinstance(value, str) and value.strip():
                    reviews.append(value.strip())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            reviews.append(item.strip())
            elif isinstance(value, (dict, list)):
                reviews.extend(extract_reviews(value))
    elif isinstance(data, list):
        for item in data:
            reviews.extend(extract_reviews(item))
    return reviews

def run_agent_with_retries(agent, task, retries=3, delay=2):
    for attempt in range(1, retries + 1):
        try:
            return asyncio.run(agent.run(task=task))
        except Exception as e:
            logging.error("Agent failed on attempt %d: %s", attempt, e)
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    return None  # Return None if all attempts fail

# Paths for input and output JSON files.
input_file_path = 'sample.json'
output_file_path = 'new_sample_output_contradiction.json'  # Save output to a new file

# Load the JSON file.
try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
except Exception as e:
    print(f"Failed to read file: {e}")
    sys.exit(1)

# Dictionary to store reviews for each paper by paper_id.
reviews_by_paper = {}

if isinstance(papers, list):
    for paper in papers:
        paper_id = paper.get("paper_id")
        if not paper_id:
            logging.warning("A paper does not have a 'paper_id' key. Skipping it.")
            continue
        reviews_by_paper[paper_id] = extract_reviews(paper)
else:
    for paper_id, paper in papers.items():
        reviews_by_paper[paper_id] = extract_reviews(paper)
# print(reviews_by_paper)
# exit(0)

# Initialize the model client and extractor agent with an updated system message.
extractor_system_message = (
    "You are a research scientist." # "You are a research scientist expert in the field of topic of the paper. Your response must be in valid JSON format and nothing else. "
    "Always respond using the JSON format exactly as specified. If no contradictions are found between any two reviews, output [] (an empty JSON list)."
)
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", response_format=["JSON"], api_key=API_KEY, base_url = "https://api.openai.com/v1")
extractor_agent = AssistantAgent("ExtractorAgent", model_client, system_message=extractor_system_message)

# --- Extraction Phase ---
# Updated extraction prompt with instructions to provide aspect_score and evidence_score (1 to 5)

## OLD prompt

# extraction_prompt = (
#     "Analyze the following academic paper reviews pairwise and identify any contradictions between two specific reviews. "
#     "A contradiction means that a statement in one review directly conflicts with a statement in another review.\n\n"
#     "Focus on these key aspects, using the following definitions:\n"
#     "1) Substance: Does the paper contain substantial experiments to demonstrate the effectiveness of proposed methods? "
#     "Are there detailed result analyses? Does it contain meaningful ablation studies?\n"
#     "2) Motivation: Does the paper address an important problem? Are other people (practitioners or researchers) likely "
#     "to use these ideas or build on them?\n"
#     "3) Clarity: For a reasonably well-prepared reader, is it clear what was done and why? "
#     "Is the paper well-written and well-structured?\n"
#     "4) Meaningful comparison: Are the comparisons to prior work sufficient given the space constraints? "
#     "Are the comparisons fair?\n"
#     "5) Originality: Are there new research topics, techniques, methodologies, or insights?\n"
#     "6) Soundness: Is the proposed approach sound? Are the claims in the paper convincingly supported?\n"
#     "7) Replicability: Is it easy to reproduce the results and verify the correctness of the results? "
#     "Is there supporting dataset and/or software provided?\n\n"
#     "For each pair of reviews (e.g., Review 1 vs Review 2) that contain a contradiction:\n"
#     "- Summarize the contradiction in 'contradiction_statement'.\n"
#     "- Identify the relevant aspect in 'identified_aspect'.\n"
#     "- Include the conflicting sentences in 'evidence' as follows: 'Review X: [complete sentence]. Review Y: [complete sentence].'\n"
#     "- Provide 'aspect_score' (an integer from 1 to 5) indicating how strongly this contradiction pertains to that aspect.\n"
#     "- Provide 'evidence_score' (an integer from 1 to 5) indicating how convincing/clear the evidence is.\n\n"
#     "If no contradictions are found between any pair of reviews, respond with an empty JSON list: []\n"
#     "Respond ONLY with a JSON list."
# )

## NEW prompt

# extraction_prompt = (
#     "Step 1: Read and analyze the following academic paper reviews pairwise.\n"
#     "Step 2: Identify any contradictions between two specific reviews. A contradiction means that a statement in one review directly conflicts with a statement in another review.\n"
#     "Step 3: Before giving your final answer, perform detailed chain-of-thought reasoning internally. Analyze all contradictions step-by-step while strictly referring to the definitions provided below. DO NOT include any of this internal reasoning in your final output.\n"
#     "Step 4: Use the following definitions to determine the relevant aspect for each contradiction:\n"
#     "    - Substance: The paper lacks substantial experiments or detailed analyses (e.g., insufficient experiments, poor result analysis, missing ablation studies).\n"
#     "    - Motivation: The paper fails to address an important problem or its significance is questionable (e.g., the research lacks impact or relevance).\n"
#     "    - Clarity: The paper is poorly written, unorganized, or unclear about its contributions and methodology.\n"
#     "    - Meaningful comparison: The paper does not fairly compare its methods with prior work or omits necessary comparative analysis.\n"
#     "    - Originality: The paper does not offer new research topics, techniques, or insights, or its contributions are incremental.\n"
#     "    - Soundness: The paper's methodology or claims are not convincingly supported or are logically inconsistent.\n"
#     "    - Replicability: The paper does not provide sufficient details, data, or code for others to reproduce its results.\n"
#     "Step 5: For each pair of reviews (e.g., Review 1 vs Review 2) that contain a contradiction, do the following:\n"
#     "    - Summarize the contradiction in 'contradiction_statement' concisely.\n"
#     "    - Identify and clearly specify the most relevant aspect (choose exactly one from: Substance, Motivation, Clarity, Meaningful comparison, Originality, Soundness, Replicability) in 'identified_aspect' by comparing the contradiction against the definitions above.\n"
#     "    - Extract the exact complete sentences from each review that illustrate the contradiction, and include them in 'evidence' formatted as: 'Review X: [complete sentence]. Review Y: [complete sentence].' Do not paraphrase or alter the sentences.\n"
#     "    - Provide an 'aspect_score' (an integer from 1 to 5) that reflects how strongly this contradiction pertains to the identified aspect (with 5 being very severe).\n"
#     "    - Provide an 'evidence_score' (an integer from 1 to 5) that indicates how clear and convincing the extracted evidence is (with 5 being very clear and directly supporting the contradiction).\n"
#     "Step 6: If no contradictions are found between any pair of reviews, respond with an empty JSON list: []\n"
#     "Step 7: Respond ONLY with a JSON list containing your final answers. Do NOT include any of your internal chain-of-thought reasoning."
# )

extraction_prompt = (
    "Step 1: Read and analyze the following academic paper reviews pairwise.\n"
    "Step 2: Identify any contradictions between two specific reviews. A contradiction means that a statement in one review directly conflicts with a statement in another review.\n"
    "Step 3: Before giving your final answer, perform detailed chain-of-thought reasoning internally. Analyze all contradictions step-by-step while strictly referring to the definitions provided below. DO NOT include any of this internal reasoning in your final output.\n"
    "Step 4: Use the following unambiguous definitions to determine the relevant aspect for each contradiction:\n"
    "    - Substance: The paper lacks substantial experiments or detailed analyses (e.g., insufficient experiments, poor result analysis, missing ablation studies).\n"
    "    - Motivation: The paper fails to address an important problem or its significance is questionable (e.g., the research lacks impact or relevance).\n"
    "    - Clarity: The paper is poorly written, unorganized, or unclear about its contributions and methodology.\n"
    "    - Meaningful comparison: The paper does not fairly compare its methods with prior work or omits necessary comparative analysis.\n"
    "    - Originality: The paper does not offer new research topics, techniques, or insights, or its contributions are incremental.\n"
    "    - Soundness: The paper's methodology or claims are not convincingly supported or are logically inconsistent.\n"
    "    - Replicability: The paper does not provide sufficient details, data, or code for others to reproduce its results.\n"
    "Step 5: For each pair of reviews (e.g., Review 1 vs Review 2) that contain a contradiction, do the following:\n"
    "    - Summarize the contradiction in a field called 'contradiction_statement' concisely.\n"
    "    - Identify and clearly specify the most relevant aspect in a field called 'identified_aspect'. Choose exactly one from: Substance, Motivation, Clarity, Meaningful comparison, Originality, Soundness, or Replicability.\n"
    "    - Extract the exact complete sentences from each review that illustrate the contradiction and include them in a field called 'evidence' formatted as: 'Review X: [complete sentence]. Review Y: [complete sentence].' Do not paraphrase or alter the sentences.\n"
    "    - Provide an 'aspect_score' (an integer from 1 to 5) that reflects how strongly this contradiction pertains to the identified aspect (with 5 being very severe).\n"
    "    - Provide an 'evidence_score' (an integer from 1 to 5) that indicates how clear and convincing the extracted evidence is (with 5 being very clear and directly supporting the contradiction).\n"
    "Step 6: If no contradictions are found between any pair of reviews, respond with an empty JSON list: [].\n"
    "Step 7: Respond ONLY with a JSON list containing your final answers. Do NOT include any of your internal chain-of-thought reasoning.\n\n"
    "Examples:\n"
    "Example 1:\n"
    "Review 1: \"The experiments are minimal and lack statistical significance.\"\n"
    "Review 2: \"The paper provides extensive experiments with detailed analysis.\"\n"
    "Contradiction Statement: \"Review 1 criticizes the experimental design while Review 2 praises its comprehensiveness.\"\n"
    "Identified Aspect: Substance\n"
    "Aspect Score: 5\n"
    "Evidence Score: 5\n\n"
    "Example 2:\n"
    "Review 1: \"The methodology is confusing and poorly explained.\"\n"
    "Review 2: \"The approach is logically structured and well justified.\"\n"
    "Contradiction Statement: \"Review 1 finds the methodology confusing, whereas Review 2 considers it logically sound.\"\n"
    "Identified Aspect: Clarity\n"
    "Aspect Score: 4\n"
    "Evidence Score: 4\n\n"
    "Remember: Perform your internal chain-of-thought analysis without outputting any of that reasoning in your final response."
)
logging.debug("Extraction prompt:\n%s", extraction_prompt)

# For each paper, combine reviews and run extraction to get contradictions with evidence.
for paper_id, reviews in reviews_by_paper.items():
    print(f"\nProcessing Paper ID: {paper_id}")
    
    if not reviews:
        print(f"No reviews found for Paper ID: {paper_id}. Saving empty agent_response.")
        if isinstance(papers, list):
            for paper in papers:
                if paper.get("paper_id") == paper_id:
                    paper["agent_response"] = []
                    break
        else:
            papers[paper_id]["agent_response"] = []
        continue
    
    combined_reviews = ""
    for idx, review in enumerate(reviews, start=1):
        combined_reviews += f"Review {idx}: {review}\n\n"
    logging.debug("Combined reviews for Paper ID %s:\n%s", paper_id, combined_reviews)
    
    # Send prompt + combined reviews to the LLM
    extraction_result = run_agent_with_retries(
        extractor_agent,
        extraction_prompt + "\n" + combined_reviews
    )
    
    # print("extraction result:",extraction_result)
    # exit(0)
    extraction_text = ""
    if extraction_result is not None and hasattr(extraction_result, "messages"):
        for msg in extraction_result.messages:
            if hasattr(msg, "source") and msg.source == "ExtractorAgent":
                extraction_text = msg.content
                break
    if not extraction_text:
        if extraction_result is not None and hasattr(extraction_result, "get_text"):
            extraction_text = extraction_result.get_text()
        else:
            extraction_text = str(extraction_result)
    logging.debug("Raw extraction output for Paper ID %s: %s", paper_id, extraction_text)
    print("Raw extraction output:", extraction_text)
    exit(0)
    
    if not extraction_text.strip():
        extraction_text = "[]"
    
    try:
        contradictions_list = json.loads(extraction_text)
    except json.JSONDecodeError as e:
        logging.error("JSON decoding error for Paper ID %s: %s", paper_id, e)
        # Attempt to extract valid JSON substring
        start = extraction_text.find('[')
        end = extraction_text.rfind(']')
        json_str = extraction_text[start:end+1] if start != -1 and end != -1 else ""
        try:
            contradictions_list = json.loads(json_str)
        except Exception as e:
            logging.error("Error parsing extraction JSON substring for Paper ID %s: %s", paper_id, e)
            contradictions_list = []
    
    logging.debug("Extracted Contradictions for Paper ID %s: %s", paper_id, contradictions_list)
    print("Extracted Contradictions:", contradictions_list)
    
    # Create a new list that contains only the desired keys from each extracted contradiction.
    extraction_results = []
    for contradiction in contradictions_list:
        stmt = contradiction.get("contradiction_statement", "")
        aspect = contradiction.get("identified_aspect", "")
        evidence = contradiction.get("evidence", "")
        
        # Attempt to parse scores as integers if they exist
        # If the model doesn't provide them or returns invalid, fallback to None
        try:
            aspect_score = int(contradiction.get("aspect_score", None))
        except (ValueError, TypeError):
            aspect_score = None
        
        try:
            evidence_score = int(contradiction.get("evidence_score", None))
        except (ValueError, TypeError):
            evidence_score = None
        
        extraction_results.append({
            "contradiction_statement": stmt,
            "identified_aspect": aspect,
            "evidence": evidence,
            "aspect_score": aspect_score,
            "evidence_score": evidence_score
        })
    
    # Save extraction results directly in the paper's agent_response field
    if isinstance(papers, list):
        for paper in papers:
            if paper.get("paper_id") == paper_id:
                paper["agent_response"] = extraction_results
                break
    else:
        papers[paper_id]["agent_response"] = extraction_results

    print(f"Extraction complete for Paper ID: {paper_id}. Results saved under 'agent_response'.")

# Finally, write out the updated JSON with contradictions + scores to the output file.
try:
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=4)
    logging.info("Extraction complete. Results saved in %s", output_file_path)
    print(f"\nExtraction complete. Results saved in {output_file_path}.")
except Exception as e:
    logging.error("Failed to write output to file: %s", e)
    raise RuntimeError(f"Failed to write output to file: {e}")