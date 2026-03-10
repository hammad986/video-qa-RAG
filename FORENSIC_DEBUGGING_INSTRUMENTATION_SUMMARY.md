# FORENSIC DEBUGGING INSTRUMENTATION - SUMMARY

## What Was Added

Comprehensive diagnostic logging has been added to the VIDEO QA RAG system to trace where valid answers are being rejected in the pipeline.

**NO CODE LOGIC WAS CHANGED** — only logging statements were added.

---

## Files Modified

### 1. **`video_qa/retrieval.py`**
- Added detailed retrieval metrics logging
- Logs each chunk's score, timestamp, and text preview

### 2. **`video_qa/answer_generator.py`** (Major instrumentation)
- **Stage 2**: Context passing debug logs
- **Stage 2B**: Prompt inspection logs  
- **Stage 3**: LLM Pass 1 response capture
- **Stage 4**: Semantic fallback trigger detection
- **Stage 4B**: LLM Pass 2 response capture
- **Stage 5**: Verification before/after comparison
- **Stage 5C**: Answer modification tracking
- **Stage 6**: Final output logging (what user sees)

### 3. **`video_qa/evidence_verifier.py`**
- **Verifier Entry**: Input validation logging
- **Path Selection**: NLI vs Ollama vs Fallback decision logging
- **Path Details**: Each verification method's result logging

---

## Debug Output Format

All debug logs are marked with `[DEBUG]` tags for easy filtering:

```bash
# View only debug logs
grep "\[.*DEBUG\]" video_qa.log

# View full logs with debug
tail -f video_qa.log
```

---

## Key Debug Sections

### **Section 1: RETRIEVAL**
```
[RETRIEVAL DEBUG] Total chunks retrieved: X
  [Chunk 1] Score: Y | Timestamp: Zs-Zs | Text: "..."
```
**Location**: Lines 50-70 in retrieval.py

### **Section 2: CONTEXT PASSING**
```
[CONTEXT DEBUG]
  contexts_passed: X
  combined_context_length: Y characters
```
**Location**: Lines ~620-640 in answer_generator.py

### **Section 3: PROMPT**
```
[PROMPT DEBUG]
  prompt_length: X characters
  prompt_preview: "..."
```
**Location**: Lines ~645-650 in answer_generator.py

### **Section 4: LLM PASS 1**
```
[LLM RESPONSE DEBUG - PASS 1]
  raw_response_length: X characters
  parsed_answer: "..."
  _is_not_found(answer): True/False
```
**Location**: Lines ~655-670 in answer_generator.py

### **Section 5: PASS 2 TRIGGER**
```
[PASS DEBUG]
  pass_1_result: NOT_FOUND
  pass_2_triggered: True/False
```
**Location**: Lines ~673-680 in answer_generator.py

### **Section 6: LLM PASS 2** (if triggered)
```
[LLM RESPONSE DEBUG - PASS 2]
  raw_response_length: X characters
  parsed_answer: "..."
  _is_not_found(sem_answer): True/False
```
**Location**: Lines ~685-700 in answer_generator.py

### **Section 7: VERIFICATION BEFORE**
```
[VERIFICATION DEBUG - BEFORE]
  answer_before_verification: "..."
  contexts_for_verification: X
```
**Location**: Lines ~703-718 in answer_generator.py

### **Section 8: VERIFIER PATHS**
```
[VERIFICATION PATH 1 - NLI]
  nli_status: VERIFIED|INCOMPLETE|HALLUCINATION|UNKNOWN
  nli_trust_score: X
  is_valid: True/False
```
**Location**: Lines ~220-260 in evidence_verifier.py

### **Section 9: VERIFICATION DECISION**
```
[VERIFICATION] Answer REJECTED/ACCEPTED by verification
  reason: ...
```
**Location**: Lines ~710-730 in answer_generator.py

### **Section 10: FINAL OUTPUT**
```
[FINAL OUTPUT - RETURNING TO UI]
  FINAL_ANSWER: "..."
  FINAL_TIMESTAMP: "..."
  VERIFICATION_STATUS: ...
  NUM_CONTEXTS: X
```
**Location**: Lines ~740-750 in answer_generator.py

---

## How to Use

### **Quick Start**
1. Open two terminals
2. **Terminal 1**: Run the system
   ```bash
   python -m streamlit run video_qa\app.py
   ```
3. **Terminal 2**: Monitor logs
   ```bash
   tail -f video_qa.log | grep "\[.*DEBUG\]"
   ```
4. Ask a question in Streamlit UI
5. Watch the debug output flow through all stages

### **The 10-Step Diagnostic**
1. Check `[RETRIEVAL DEBUG]` - Did retrieval work?
2. Check `[CONTEXT DEBUG]` - Were contexts passed?
3. Check `[LLM RESPONSE DEBUG - PASS 1]` - What did LLM generate?
4. Check `[PASS DEBUG]` - Did Pass 2 trigger?
5. Check `[LLM RESPONSE DEBUG - PASS 2]` - Did fallback work?
6. Check `[VERIFICATION ENTRY]` - What inputs to verifier?
7. Check `[VERIFICATION PATH]` - Which method used?
8. Check `[VERIFICATION] status & is_valid` - Was answer rejected?
9. Check `[FINAL OUTPUT]` - What was returned to UI?
10. **Compare**: Did NUM_CONTEXTS > 0 but FINAL_ANSWER = "NOT FOUND"?

### **Example: Finding the Root Cause**

**Scenario**: User sees "NOT FOUND IN VIDEO" but transcript clearly contains the answer

**Diagnostic output:**
```
✅ [RETRIEVAL DEBUG] chunks_retrieved: 5
✅ [CONTEXT DEBUG] contexts_passed: 5
✅ [LLM RESPONSE DEBUG - PASS 1] parsed_answer: "Regularization is..."
✅ [PASS DEBUG] pass_2_triggered: False (not needed, Pass 1 succeeded)
❌ [VERIFICATION] status: INCOMPLETE, is_valid: False
❌ [FINAL OUTPUT] FINAL_ANSWER: "NOT FOUND IN VIDEO"
```

**Conclusion**: Verification is too strict. It's rejecting valid answers.

---

## What Each Metric Tells You

| Metric | Expected | Problem |
|--------|----------|---------|
| `chunks_retrieved > 0` | Yes | Retrieval failed |
| `contexts_passed > 0` | Yes | **Contexts were dropped!** |
| `LLM Pass 1 _is_not_found` | False | LLM strict pass failed |
| `pass_2_triggered` | True (if Pass 1 failed) | Fallback didn't trigger |
| `LLM Pass 2 _is_not_found` | False | Both passes too strict |
| `verifier nli_status` | VERIFIED | Verification accepted |
| `verification is_valid` | True | Should return answer |
| `FINAL_ANSWER` | Actual answer | "NOT FOUND IN VIDEO" = rejected |

---

## Critical Checkpoints

### **If NUM_CONTEXTS > 0 but FINAL_ANSWER = "NOT FOUND IN VIDEO"**

The answer was rejected somewhere in the pipeline:

1. **Possibility 1**: LLM couldn't extract → Check Pass 1 & 2 responses
2. **Possibility 2**: Semantic fallback didn't trigger → Check Pass 2 trigger condition
3. **Possibility 3**: Verification rejected it → Check verification status
4. **Possibility 4**: Answer was modified → Check [ANSWER MODIFICATION DEBUG]

---

## Common Issues to Look For

### **Issue 1: Contexts Passed = 0**
```
[CONTEXT DEBUG]
  contexts_passed: 0
```
**Problem**: Contexts were dropped between retrieval and answer generator
**Check**: Is there a bug in the pipeline flow?

### **Issue 2: Both Passes Failed**
```
[LLM RESPONSE DEBUG - PASS 1] _is_not_found: True
[LLM RESPONSE DEBUG - PASS 2] _is_not_found: True
```
**Problem**: LLM prompts are too strict
**Check**: Prompt engineering in build_answer_prompt() and build_semantic_prompt()

### **Issue 3: Verification Rejected Valid Answer**
```
[VERIFICATION] status: INCOMPLETE, is_valid: False
[FINAL OUTPUT] FINAL_ANSWER: "NOT FOUND IN VIDEO"
```
**Problem**: Verification is rejecting valid grounded answers
**Check**: NLI model calibration, threshold settings

### **Issue 4: Answer Modified After Verification**
```
[VERIFICATION DEBUG - BEFORE] answer: "Regularization is..."
[ANSWER MODIFICATION DEBUG] answer: "Question cannot be answered..."
```
**Problem**: Verification modified the answer
**Check**: is_valid logic in answer_generator.py

---

## Next Steps with Debug Information

1. **Run a failing query** (e.g., "What is regularization?")
2. **Capture the debug output**
   ```bash
   grep "\[.*DEBUG\]" video_qa.log > debug_output.txt
   ```
3. **Identify the failure point** using the 10-step diagnostic
4. **Document findings** in a bug report
5. **Apply targeted fix** to that specific stage

---

## Documentation Files

Three documentation files have been created:

1. **`DEBUG_INSTRUMENTATION.md`** (this directory)
   - Complete reference for all debug sections
   - Expected behavior examples
   - How to follow the pipeline

2. **`DEBUG_QUICK_REFERENCE.md`** (this directory)
   - Quick checkpoint list
   - Decision tree for root cause
   - Example debug session

3. **`FORENSIC_DEBUGGING_INSTRUMENTATION_SUMMARY.md`** (your reading now)
   - Overview of what was added
   - How to use the instrumentation
   - Common issues

---

## Important Notes

### ✅ What Was Changed
- Added ~200 lines of logger.info() statements
- No core logic modified
- No function signatures changed
- Fully backward compatible

### ❌ What Was NOT Changed
- Answer generation logic
- Verification logic
- Retrieval logic
- Any decision-making code

### 🎯 Purpose
- **Diagnose** - Locate exactly where answers are rejected
- **Not Fix** - These changes don't fix the underlying issue
- **Transparent** - All logs are non-invasive

---

## Troubleshooting

### **Q: I don't see any DEBUG logs**
**A**: 
1. Check log level in config.yaml
2. Ensure logger is initialized
3. Verify log file path is correct
4. Try: `tail -f video_qa.log` without grep first

### **Q: Logs are too verbose**
**A**:
```bash
grep "\[FINAL OUTPUT\]" video_qa.log
```
Or grep specific stage:
```bash
grep "\[VERIFICATION\]" video_qa.log
```

### **Q: Where is the log file?**
**A**: Check config.yaml for `logger.log_file` setting. Default is `video_qa.log` in project root.

### **Q: Can I disable these logs?**
**A**: Find lines with `logger.info("[" in answer_generator.py and comment them out. Or set log level to WARNING in config.yaml.

---

## Summary

This instrumentation adds **comprehensive diagnostic capabilities** without changing any system logic. Use it to:

1. ✅ **Trace** the pipeline stage-by-stage
2. ✅ **Identify** exactly where answers are rejected
3. ✅ **Understand** each decision point in the verification process
4. ✅ **Report** precise issues with supporting logs
5. ✅ **Fix** only the broken stage

The instrumentation is complete and ready for use! 🔍

