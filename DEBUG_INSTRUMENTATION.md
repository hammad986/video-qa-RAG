# VIDEO QA RAG - Forensic Debugging Instrumentation Guide

## Overview

Comprehensive logging instrumentation has been added to locate **exactly where valid answers are being rejected** in the pipeline.

**To view debug logs:**
```bash
tail -f video_qa.log | grep "\[.*DEBUG\]"
```

---

## Pipeline Stages & Debug Points

### **STAGE 1: RETRIEVAL** 
**File:** `retrieval.py` (lines ~50-70)

**What it logs:**
```
[RETRIEVAL DEBUG]
- Total chunks retrieved
- For each chunk:
  - Score (similarity)
  - Timestamp range
  - Text preview (first 80 chars)
```

**Example output:**
```
[RETRIEVAL DEBUG] Total chunks retrieved: 5
  [Chunk 1] Score: 0.7234 | Timestamp: 120.5s-140.2s | Text: "Regularization is a technique..."
  [Chunk 2] Score: 0.6891 | Timestamp: 155.0s-175.1s | Text: "In machine learning, overfitting..."
  [Chunk 3] Score: 0.5423 | Timestamp: 200.2s-220.5s | Text: "The purpose of regularization..."
```

**What it reveals:**
- ✅ Are chunks being retrieved?
- ✅ What are the similarity scores?
- ✅ Do retrieved chunks answer the question?

---

### **STAGE 2: CONTEXT PASSING**
**File:** `answer_generator.py` (lines ~620-640)

**What it logs:**
```
[CONTEXT DEBUG]
- Number of contexts passed to LLM
- Combined context length (total characters)
- For each context:
  - Timestamp range
  - Text length
  - Score
```

**Example output:**
```
[CONTEXT DEBUG]
  contexts_passed: 3
  combined_context_length: 2847 characters
    Context[1]: 120.5s-140.2s | Length: 856 chars | Score: 0.7234
    Context[2]: 155.0s-175.1s | Length: 923 chars | Score: 0.6891
    Context[3]: 200.2s-220.5s | Length: 1068 chars | Score: 0.5423
```

**What it reveals:**
- ✅ Are contexts actually passed to the LLM?
- ✅ How much context is the LLM seeing?
- ⚠️ If `contexts_passed: 0`, contexts were dropped

---

### **STAGE 2B: PROMPT INSPECTION**
**File:** `answer_generator.py` (lines ~645-650)

**What it logs:**
```
[PROMPT DEBUG]
- Prompt length (total characters)
- Prompt preview (first 300 chars)
```

**Example output:**
```
[PROMPT DEBUG]
  prompt_length: 2856 characters
  prompt_preview: "You are a VIDEO QUESTION ANSWERING SYSTEM.

Rules:
* Only use the provided evidence passages...
```

**What it reveals:**
- ✅ Is the LLM receiving the full context?
- ✅ Does the prompt align with retrieved contexts?

---

### **STAGE 3: LLM RESPONSE (PASS 1 - STRICT)**
**File:** `answer_generator.py` (lines ~655-670)

**What it logs:**
```
[LLM RESPONSE DEBUG - PASS 1]
- Raw response length
- Raw response (first 300 chars)
- Parsed answer
- Parsed timestamp  
- _is_not_found verdict (True/False)
```

**Example output (GOOD - LLM found answer):**
```
[LLM RESPONSE DEBUG - PASS 1]
  raw_response_length: 234 characters
  raw_response (first 300 chars): "Answer: Regularization helps reduce overfitting
Timestamp: [02:00 - 02:15]"
  parsed_answer: "Regularization helps reduce overfitting"
  parsed_timestamp: [02:00 - 02:15]
  _is_not_found(answer): False
```

**Example output (BAD - LLM says not found):**
```
[LLM RESPONSE DEBUG - PASS 1]
  raw_response_length: 89 characters
  raw_response (first 300 chars): "Answer: NOT FOUND IN VIDEO
Timestamp: None"
  parsed_answer: "NOT FOUND IN VIDEO"
  parsed_timestamp: None
  _is_not_found(answer): True
```

**What it reveals:**
- ❌ **CRITICAL**: Is the strict LLM returning "NOT FOUND" when it shouldn't?
- ✅ Is the LLM actually using the context?
- ✅ Can it extract a proper timestamp?

---

### **STAGE 4: PASS 2 TRIGGER (SEMANTIC FALLBACK)**
**File:** `answer_generator.py` (lines ~673-680)

**What it logs:**
```
[PASS DEBUG]
- pass_1_result: NOT_FOUND or SUCCESS
- pass_1_answer: the actual answer text
- pass_2_triggered: True/False (should be True if Pass 1 failed)
- reason_for_pass_2: explanation
```

**Example output (Pass 2 triggered):**
```
[PASS DEBUG]
  pass_1_result: NOT_FOUND
  pass_1_answer: "NOT FOUND IN VIDEO"
  pass_2_triggered: True
  reason_for_pass_2: strict returned NOT_FOUND AND 3 contexts exist
```

**Example output (Pass 2 NOT triggered - meaning Pass 1 succeeded):**
```
[LLM RESPONSE DEBUG - PASS 1]
  ...
  _is_not_found(answer): False   <--- Pass 2 won't trigger
```

**What it reveals:**
- ✅ When does Pass 2 trigger?
- ✅ Is there a fallback mechanism for strict pass failures?

---

### **STAGE 4B: PASS 2 RESPONSE (SEMANTIC MODE)**
**File:** `answer_generator.py` (lines ~685-700)

**What it logs:**
```
[LLM RESPONSE DEBUG - PASS 2]
- Raw response length
- Raw response (first 300 chars)
- Parsed answer
- _is_not_found verdict
```

**Example output (Pass 2 succeeds):**
```
[LLM RESPONSE DEBUG - PASS 2]
  raw_response_length: 167 characters
  raw_response (first 300 chars): "Answer: Regularization is used to prevent overfitting
Timestamp: [02:00 - 02:15]"
  parsed_answer: "Regularization is used to prevent overfitting ..."
  _is_not_found(sem_answer): False
```

**Example output (Pass 2 also fails):**
```
[LLM RESPONSE DEBUG - PASS 2]
  raw_response_length: 89 characters
  raw_response (first 300 chars): "Answer: NOT FOUND IN VIDEO
Timestamp: None"
  parsed_answer: "NOT FOUND IN VIDEO"
  _is_not_found(sem_answer): True
```

**What it reveals:**
- ✅ Can Pass 2 recover from Pass 1 failure?
- ❌ **ROOT CAUSE**: If Pass 2 also fails, the semantic LLM is too strict

---

### **STAGE 5: VERIFICATION (BEFORE)**
**File:** `answer_generator.py` (lines ~703-718)

**What it logs:**
```
[VERIFICATION DEBUG - BEFORE]
- Answer before verification
- Timestamp before verification
- Number of contexts for verification
- Message: "running verification..."
```

**Example output:**
```
[VERIFICATION DEBUG - BEFORE]
  answer_before_verification: "Regularization is used to prevent overfitting..."
  timestamp_before_verification: [02:00 - 02:15]
  contexts_for_verification: 3
  running verification...
```

**What it reveals:**
- ✅ What answer is being sent to verification?
- ✅ How many contexts is verification receiving?

---

### **STAGE 5A: VERIFIER ENTRY POINT**
**File:** `evidence_verifier.py` (lines ~182-198)

**What it logs:**
```
[VERIFIER ENTRY]
- Answer input (first 100 chars)
- Contexts count
- Query (first 80 chars)
```

**Example output:**
```
[VERIFIER ENTRY]
  answer (first 100 chars): "Regularization is used to prevent overfitting..."
  contexts_count: 3
  query: "What is regularization..."
```

**What it reveals:**
- ✅ Is verification receiving the correct inputs?

---

### **STAGE 5B: VERIFICATION PATHS (NLI → OLLAMA → FALLBACK)**
**File:** `evidence_verifier.py` (lines ~220-260)

**What it logs:**
```
[VERIFICATION PATH 1 - NLI]
- nli_status: VERIFIED | INCOMPLETE | HALLUCINATION | UNKNOWN
- nli_trust_score: 0-100
- nli_confidence: 0.0-1.0
- Decision: "Returning NLI result" or "trying Ollama..."

[VERIFICATION PATH 2 - OLLAMA]
- ollama_status: (if available)
- Decision: "Returning Ollama result" or proceeding to Path 3

[VERIFICATION PATH 3 - CONSERVATIVE FALLBACK]
- Decision: "Accepting answer conservatively"
```

**Example output (NLI accepts):**
```
[VERIFICATION PATH 1 - NLI]
  nli_status: VERIFIED
  nli_trust_score: 90
  nli_confidence: 90.00%
  -> Returning NLI result: is_valid=True
```

**Example output (NLI rejects):**
```
[VERIFICATION PATH 1 - NLI]
  nli_status: HALLUCINATION
  nli_trust_score: 10
  nli_confidence: 10.00%
  -> Returning NLI result: is_valid=False
```

**Example output (NLI uncertain):**
```
[VERIFICATION PATH 1 - NLI]
  nli_status: INCOMPLETE
  nli_trust_score: 30
  nli_confidence: 30.00%
  -> Returning NLI result: is_valid=False
```

**What it reveals:**
- ❌ **MAJOR ROOT CAUSE**: Is NLI marking valid answers as INCOMPLETE/HALLUCINATION?
- ✅ What verification method is being used?
- ✅ What confidence is assigned?

---

### **STAGE 5C: VERIFICATION RESULT & MODIFICATION**
**File:** `answer_generator.py` (lines ~710-730)

**What it logs:**
```
[VERIFICATION DEBUG - AFTER]
- Verification status (VERIFIED | INCOMPLETE | HALLUCINATION | UNKNOWN)
- is_valid: True/False
- trust_score: 0-100
- confidence: percentage

[VERIFICATION] Answer REJECTED or ACCEPTED
- If rejected: original_answer + reason
- If accepted: confirmation
```

**Example output (REJECTED - HALLUCINATION):**
```
[VERIFICATION DEBUG - AFTER]
  status: HALLUCINATION
  is_valid: False
  trust_score: 10
  confidence: 10.00%

[VERIFICATION] Answer REJECTED by verification (status=HALLUCINATION)
  original_answer: "Regularization is used to prevent..."
  reason: HALLUCINATION - answer contradicts evidence
```

**Example output (REJECTED - INCOMPLETE):**
```
[VERIFICATION DEBUG - AFTER]
  status: INCOMPLETE
  is_valid: False
  trust_score: 30
  confidence: 30.00%

[VERIFICATION] Answer REJECTED by verification (status=INCOMPLETE)
  original_answer: "Regularization is used to prevent..."
  reason: is_valid=False (status=INCOMPLETE)
```

**Example output (ACCEPTED):**
```
[VERIFICATION DEBUG - AFTER]
  status: VERIFIED
  is_valid: True
  trust_score: 90
  confidence: 90.00%

[VERIFICATION] Answer ACCEPTED by verification (status=VERIFIED)
```

**What it reveals:**
- ❌ **CRITICAL**: Is verification rejecting valid answers?
- ❌ Which answers are being modified by verification?
- ✅ What confidence is the verifier confident?

---

### **STAGE 5D: ANSWER MODIFICATION AFTER VERIFICATION**
**File:** `answer_generator.py` (lines ~732-738)

**What it logs:**
```
[ANSWER MODIFICATION DEBUG]
- Answer after verification (may differ from before)
- Timestamp after verification
- Verified flag
```

**Example output (answer unchanged):**
```
[ANSWER MODIFICATION DEBUG]
  answer_after_verification: "Regularization is used to prevent..."
  timestamp_after_verification: [02:00 - 02:15]
  verified_flag: True
```

**Example output (answer replaced due to rejection):**
```
[ANSWER MODIFICATION DEBUG]
  answer_after_verification: "Question cannot be answered from this video."
  timestamp_after_verification: None
  verified_flag: False
```

**What it reveals:**
- ❌ **ROOT CAUSE**: Did verification replace the answer?
- ❌ Which answers got overwritten?

---

### **STAGE 6: FINAL OUTPUT (BEFORE RETURNING TO UI)**
**File:** `answer_generator.py` (lines ~740-750)

**What it logs:**
```
[FINAL OUTPUT - RETURNING TO UI]
- FINAL_ANSWER: The exact answer shown to user
- FINAL_TIMESTAMP: The timestamp shown to user
- VERIFIED: Whether marked as verified
- TRUST_SCORE: Confidence percentage
- NUM_CONTEXTS: How many context chunks
- VERIFICATION_STATUS: The verification status
```

**Example output (SUCCESS - valid answer returned):**
```
[FINAL OUTPUT - RETURNING TO UI]
  FINAL_ANSWER: "Regularization is a technique used to reduce overfitting"
  FINAL_TIMESTAMP: [02:00 - 02:15]
  VERIFIED: True
  TRUST_SCORE: 90
  NUM_CONTEXTS: 3
  VERIFICATION_STATUS: VERIFIED

================================================================================
```

**Example output (FAILURE - answer rejected at verification):**
```
[FINAL OUTPUT - RETURNING TO UI]
  FINAL_ANSWER: "NOT FOUND IN VIDEO"
  FINAL_TIMESTAMP: None
  VERIFIED: False
  TRUST_SCORE: 0
  NUM_CONTEXTS: 3
  VERIFICATION_STATUS: INCOMPLETE

================================================================================
```

**What it reveals:**
- ✅ **EXACT ROOT CAUSE**: This is what the user sees
- ❌ Is the final answer "NOT FOUND IN VIDEO" despite:
  - NUM_CONTEXTS > 0? (retrieval worked)
  - An answer generated? (LLM worked)
  - Dismissed by verification? (verifier rejected it)

---

## How to Use This Instrumentation

### **Step 1: Ask a question that's failing**
In the Streamlit UI, ask: "What is regularization?"

### **Step 2: Check the logs**
```bash
tail -f video_qa.log | grep "\[.*DEBUG\]"
```

### **Step 3: Follow the pipeline**
1. Check [RETRIEVAL DEBUG] - Did retrieval work?
2. Check [CONTEXT DEBUG] - Were contexts passed?
3. Check [LLM RESPONSE DEBUG - PASS 1] - What did LLM generate?
4. Check [PASS DEBUG] - Did Pass 2 trigger?
5. Check [LLM RESPONSE DEBUG - PASS 2] - Did fallback work?
6. Check [VERIFICATION DEBUG] - What verdict?
7. Check [FINAL OUTPUT] - What was returned to UI?

### **Step 4: Locate the exact failure point**

**If answer was "NOT FOUND IN VIDEO" but NUM_CONTEXTS > 0:**

- **Failure at Stage 3**: LLM is too strict → Need to fix prompts
- **Failure at Stage 4**: Semantic fallback didn't trigger → Check `_is_not_found()` logic
- **Failure at Stage 5**: Verification rejected it → Need to fix verifier
- **All stages succeeded but still wrong**: Check final_output mismatch

---

## Quick Reference: What Each Log Means

| Log | Normal Value | Problem Value | Root Cause |
|-----|--------------|---------------|-----------|
| [RETRIEVAL DEBUG] chunks_retrieved | > 0 | 0 | No relevant chunks found |
| [CONTEXT DEBUG] contexts_passed | > 0 | 0 | **CRITICAL**: Contexts dropped |
| [LLM RESPONSE DEBUG - PASS 1] is_not_found | False | True | LLM prompt too strict |
| [PASS DEBUG] pass_2_triggered | True | False | Pass 2 didn't trigger (good if Pass 1 succeeded) |
| [LLM RESPONSE DEBUG - PASS 2] is_not_found | False | True | Semantic fallback too strict |
| [VERIFICATION] status | VERIFIED | INCOMPLETE/HALLUCINATION | **ROOT CAUSE**: Verifier rejecting |
| [FINAL OUTPUT] FINAL_ANSWER | Actual answer | "NOT FOUND IN VIDEO" | Rejected at verification |

---

## Expected Behavior

### ✅ **CORRECT FLOW (answer is shown)**
```
[RETRIEVAL DEBUG] chunks_retrieved: 5
[CONTEXT DEBUG] contexts_passed: 5
[LLM RESPONSE DEBUG - PASS 1] parsed_answer: "Regularization is..."
[VERIFICATION DEBUG - AFTER] status: VERIFIED, is_valid: True
[FINAL OUTPUT] FINAL_ANSWER: "Regularization is..."
```

### ❌ **BROKEN FLOW (answer rejected)**
```
[RETRIEVAL DEBUG] chunks_retrieved: 5
[CONTEXT DEBUG] contexts_passed: 5
[LLM RESPONSE DEBUG - PASS 1] parsed_answer: "Regularization is..."
[VERIFICATION DEBUG - AFTER] status: INCOMPLETE, is_valid: False  <-- REJECTION HERE
[FINAL OUTPUT] FINAL_ANSWER: "NOT FOUND IN VIDEO"  <-- WRONG!
```

---

## Next Steps After Debugging

1. **Identify the exact stage** where the answer is being lost
2. **Note the debug output** from that stage
3. **Report findings** with the debug logs
4. **Apply targeted fix** only to that stage

The instrumentation is designed to **NOT fix anything** — only to **locate the problem**.

