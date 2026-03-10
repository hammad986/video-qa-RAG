# DEBUG OUTPUT QUICK REFERENCE

## How to Run and View Logs

### **1. Start the System**
```bash
cd d:\VIDEO_QA_RAG
my_project\Scripts\Activate.ps1
python -m streamlit run video_qa\app.py
```

### **2. View Debug Logs in Real-Time**
```bash
# In a separate terminal
tail -f video_qa.log | grep "\[.*DEBUG\]"
```

### **3. Or View Full Log**
```bash
tail -f video_qa.log
```

---

## Critical Debug Checkpoints

### **CHECKPOINT 1: Retrieval Success**
```
Look for:
[RETRIEVAL DEBUG] Total chunks retrieved: X

✅ Good: X > 0 (e.g., X = 5)
❌ Bad: X = 0 (no chunks found)
```

### **CHECKPOINT 2: Context Passing**
```
Look for:
[CONTEXT DEBUG]
  contexts_passed: X

✅ Good: X > 0 (matches retrieval count)
❌ Bad: X = 0 (contexts were dropped!)
```

### **CHECKPOINT 3: LLM Pass 1 Response**
```
Look for:
[LLM RESPONSE DEBUG - PASS 1]
  _is_not_found(answer): TRUE/FALSE

✅ Good: _is_not_found = False (LLM found answer)
❌ Bad: _is_not_found = True (LLM says not found)
```

### **CHECKPOINT 4: Semantic Fallback Trigger**
```
Look for:
[PASS DEBUG]
  pass_2_triggered: TRUE/FALSE

✅ Good: If Pass 1 failed, this should be True
❌ Bad: If Pass 1 failed AND this is False (no fallback)
```

### **CHECKPOINT 5: Verification Status**
```
Look for:
[VERIFICATION DEBUG - AFTER]
  status: VERIFIED | INCOMPLETE | HALLUCINATION | UNKNOWN
  is_valid: True/False

✅ Good: status=VERIFIED, is_valid=True
⚠️ Warning: status=INCOMPLETE, is_valid=False (might reject valid answer)
❌ Bad: status=HALLUCINATION, is_valid=False (definitely reject)
```

### **CHECKPOINT 6: Final Output**
```
Look for:
[FINAL OUTPUT - RETURNING TO UI]
  FINAL_ANSWER: ...
  NUM_CONTEXTS: X

✅ Good: FINAL_ANSWER is the actual answer, NUM_CONTEXTS > 0
❌ Bad: FINAL_ANSWER = "NOT FOUND IN VIDEO", NUM_CONTEXTS > 0
```

---

## Diagnostic Flow (Step-by-Step)

### **IF YOU SEE: "NOT FOUND IN VIDEO" BUT NUM_CONTEXTS > 0**

This means:
- ✅ Retrieval worked (found chunks)
- ✅ Answer was generated somewhere
- ❌ But something rejected it

**Follow this diagnostic:**

```
1. Check [RETRIEVAL DEBUG] total chunks
   If > 0 → retrieval OK ✅
   
2. Check [CONTEXT DEBUG] contexts_passed
   If > 0 → contexts OK ✅
   If = 0 → STOP HERE ❌ (contexts were dropped)
   
3. Check [LLM RESPONSE DEBUG - PASS 1] _is_not_found
   If False → Pass 1 found answer ✅
   If True → Pass 1 failed ⚠️
   
4. If Pass 1 failed, check [PASS DEBUG] pass_2_triggered
   If True → Semantic fallback running ✅
   If False → STOP HERE ❌ (no fallback)
   
5. If Pass 2 running, check [LLM RESPONSE DEBUG - PASS 2] _is_not_found
   If False → Pass 2 found answer ✅
   If True → Both passes failed (LLM prompt too strict) ❌
   
6. Check [VERIFICATION] status and is_valid
   If is_valid=False → VERIFICATION REJECTED IT ❌ (ROOT CAUSE)
   If is_valid=True → Should not see "NOT FOUND" error

7. Check [FINAL OUTPUT] FINAL_ANSWER
   If "NOT FOUND IN VIDEO" → Something modified answer after generation
```

---

## Root Cause Decision Tree

```
START: User sees "NOT FOUND IN VIDEO"

Does NUM_CONTEXTS > 0?
├─ NO  → Retrieval failed (not a verification issue)
└─ YES → Continue...

Does [LLM RESPONSE DEBUG - PASS 1] show parsed_answer?
├─ NO  → LLM failure / timeout
└─ YES (but _is_not_found=True) → Continue...

Does [PASS DEBUG] show pass_2_triggered=True?
├─ NO  → Semantic fallback disabled (check config)
└─ YES → Continue...

Does [LLM RESPONSE DEBUG - PASS 2] show _is_not_found=False?
├─ NO  → Both LLM passes too strict (PROMPT ISSUE)
└─ YES → Continue...

Does [VERIFICATION DEBUG - AFTER] show is_valid=True?
├─ NO  → VERIFICATION REJECTED IT 🎯 (ROOT CAUSE: Verifier too strict)
└─ YES → Check [ANSWER MODIFICATION DEBUG] for changes

ROOT CAUSE FOUND! ✅
```

---

## What to Report

When reporting the issue, include:

1. **The exact question asked**
   ```
   Question: "What is regularization?"
   ```

2. **The expected answer** (what should be returned)
   ```
   Expected: "Regularization is a technique..."
   ```

3. **The actual answer** (what the system returned)
   ```
   Actual: "NOT FOUND IN VIDEO"
   ```

4. **The full debug log output** (`grep "\[.*DEBUG\]" video_qa.log`)

5. **The root cause checkpoint** (from the decision tree above)
   ```
   ROOT CAUSE: Verification rejected valid answer
   Status: INCOMPLETE
   ```

---

## Example Debug Session

### **Setup**
```bash
cd d:\VIDEO_QA_RAG
tail -f video_qa.log | grep "\[.*DEBUG\]"
```

### **Ask Question** (in Streamlit UI)
```
User: "What is regularization?"
```

### **Watch Debug Output**
```
[RETRIEVAL DEBUG] Total chunks retrieved: 5
  [Chunk 1] Score: 0.7234 | Timestamp: 120.5s-140.2s | Text: "Regularization is a technique..."

[CONTEXT DEBUG]
  contexts_passed: 5
  combined_context_length: 2847 characters

[PROMPT DEBUG]
  prompt_length: 2856 characters

[LLM RESPONSE DEBUG - PASS 1]
  raw_response_length: 234 characters
  raw_response (first 300 chars): "Answer: Regularization helps reduce overfitting\nTimestamp: [02:00 - 02:15]"
  parsed_answer: "Regularization helps reduce overfitting"
  parsed_timestamp: [02:00 - 02:15]
  _is_not_found(answer): False

[VERIFICATION DEBUG - BEFORE]
  answer_before_verification: "Regularization helps reduce overfitting..."
  timestamp_before_verification: [02:00 - 02:15]
  contexts_for_verification: 5

[VERIFIER ENTRY]
  answer (first 100 chars): "Regularization helps reduce overfitting..."
  contexts_count: 5
  query: "What is regularization..."

[VERIFICATION PATH 1 - NLI]
  nli_status: INCOMPLETE
  nli_trust_score: 30
  nli_confidence: 30.00%
  -> Returning NLI result: is_valid=False

[VERIFICATION DEBUG - AFTER]
  status: INCOMPLETE
  is_valid: False
  trust_score: 30
  confidence: 30.00%

[VERIFICATION] Answer REJECTED by verification (status=INCOMPLETE)
  original_answer: "Regularization helps reduce overfitting..."
  reason: is_valid=False (status=INCOMPLETE)

[ANSWER MODIFICATION DEBUG]
  answer_after_verification: "Question cannot be answered from this video."
  timestamp_after_verification: None
  verified_flag: False

[FINAL OUTPUT - RETURNING TO UI]
  FINAL_ANSWER: "NOT FOUND IN VIDEO"
  FINAL_TIMESTAMP: None
  VERIFIED: False
  TRUST_SCORE: 0
  NUM_CONTEXTS: 5
  VERIFICATION_STATUS: INCOMPLETE
```

### **Analysis**
✅ Retrieval worked (5 chunks)
✅ Context passed to LLM (5 contexts)
✅ LLM generated answer (Pass 1 succeeded)
✅ Answer was grounded
❌ **ROOT CAUSE FOUND**: Verification rejected it
   - NLI returned INCOMPLETE
   - is_valid set to False
   - Answer was replaced

**Conclusion:** Verifier is too strict. NLI model is marking valid answers as INCOMPLETE.

---

## Next Steps

1. **Identify the root cause** using this guide
2. **Document the exact checkpoint** where failure occurs
3. **Report with debug logs**
4. **Apply targeted fix** only to that stage

The instrumentation should help pinpoint the exact problem in the pipeline! 🎯
