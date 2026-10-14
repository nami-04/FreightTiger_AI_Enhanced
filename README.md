# FreightTiger Smart Shipping-Cost Assistant (LLM-enabled)

## Features
- Groups shipments by route and supplied `route_type`, into Monday-Sunday weeks.
- Calculates cost per tonne-km as total weekly freight cost / total weekly tonne-km.
- Compares against up to the prior 8 observed route-weeks, strictly excluding the current week.
- Compares against same-week routes of the same `route_type`, excluding the current route.
- Numeric flags are deterministic: +10% or more vs own history **OR** +15% or more vs peers. The case brief does not prescribe numeric thresholds; these are explicit configurable engineering choices.
- Matches context notes only when route and time are compatible. A route/date-validated note can be passed to OpenRouter to draft the explanation. The LLM cannot set the flag or note ID.
- If no API key is configured, the system still runs with deterministic evidence-based explanations; it does not pretend an LLM was used.
- Exact required output columns are checked against the supplied sample CSV header.

## Setup (Windows 10)
1. Install Python 3.10+ from https://www.python.org/downloads/ and tick **Add Python to PATH**.
2. Extract this ZIP.
3. Open Command Prompt in the extracted `FreightTiger_AI_Enhanced` folder.
4. Install dependencies:
   ```bat
   python -m pip install -r requirements.txt
   ```
5. For the LLM feature, create an OpenRouter account and API key, then set it in the current Command Prompt session (do not share or commit it):
   ```bat
   set OPENROUTER_API_KEY=PASTE_YOUR_KEY_HERE
   set OPENROUTER_MODEL=openai/gpt-4o-mini
   ```
   Check current model availability/pricing in OpenRouter before selecting a model. A model may be free, paid, or change availability. Never commit the key into GitHub.
6. Run:
   ```bat
   python freight_assistant.py
   ```
7. Generate the repeat check:
   ```bat
   python freight_assistant.py --repeat-check
   ```
   Three full runs can make multiple API calls. Run without an API key for a zero-LLM-call determinism check, or keep the key and check provider usage. Even temperature 0 does not guarantee identical LLM wording across provider/model updates; numeric flags, metrics and note IDs are independent of LLM wording.

## Files
- `freight_assistant.py`: implementation and optional OpenRouter client.
- `shipment_records.csv`, `context_notes.csv`: supplied data.
- `sample_output_format_v2.csv`: schema contract/example.
- `output.csv`: generated result.
- `README.md`: instructions and assumptions.
- `token_cost_log.txt`: per-run actual provider token usage where returned.
- `reproducibility_check.txt`: hash evidence from a three-run check.
- `.env.example`: environment variable names only; no secret is included.

## AI and trust architecture
1. Pandas computes all numeric features and flags.
2. Local deterministic matching restricts notes by route and date; explicitly non-causal notes are excluded as positive support.
3. Only a validated candidate note is sent to the LLM for wording.
4. The LLM prompt says to use only supplied facts. The code checks that its answer includes the validated note ID; otherwise it falls back to a deterministic explanation.
5. If no valid note is found, the row is unexplained, `matched_note_id` is blank, and the assistant does not invent a cause.

This is a small-corpus baseline, not a production-grade semantic vector database. A vector store such as Qdrant would become useful with a much larger notes corpus, but semantic retrieval must still be followed by strict route/date validation.

## Cost accounting
The script records provider-reported prompt and completion tokens and the number of successful calls. It deliberately does not invent a price: calculate cost using the selected model's current published rates or provider usage dashboard. With no API key, model calls and model tokens are zero and explanations use the deterministic fallback.

## Known limitations to explain honestly
- Thresholds (+10% and +15%) are chosen because the brief does not set exact thresholds.
- Notes are short and sparse; matching uses deterministic route/date rules and one explicit multi-day incident window rather than embeddings.
- The LLM is optional and only drafts explanations for validated notes. It does not implement free-form natural-language Q&A.
- For the strictest reproducibility claim, compare flags, numeric columns, and note IDs separately from LLM wording.
