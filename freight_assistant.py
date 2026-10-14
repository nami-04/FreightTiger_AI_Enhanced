#!/usr/bin/env python3
"""FreightTiger Smart Shipping-Cost Assistant with optional OpenRouter LLM explanations.

Core numbers and flags are deterministic. The LLM can only explain a validated
candidate note; it never chooses the numeric anomaly flag or note ID.
"""
from __future__ import annotations
import argparse, json, os, hashlib, re, time
from pathlib import Path
import pandas as pd

OUTPUT_COLUMNS = ["route","week_of","cost_per_tonne_km","vs_own_history","vs_similar_routes","flagged","matched_note_id","reason"]
OWN_RISE_THRESHOLD_PCT = 10.0
PEER_RISE_THRESHOLD_PCT = 15.0
MODEL_DEFAULT = "openai/gpt-4o-mini"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def load_inputs(data_dir: Path):
    shipments = pd.read_csv(data_dir/"shipment_records.csv", parse_dates=["shipment_date"])
    notes = pd.read_csv(data_dir/"context_notes.csv", keep_default_na=False)
    # Read header only: supplied sample file contains illustrative rows, not a blank template.
    template_cols = pd.read_csv(data_dir/"sample_output_format_v2.csv", nrows=0).columns.tolist()
    if template_cols != OUTPUT_COLUMNS:
        raise ValueError(f"Output schema mismatch: expected {OUTPUT_COLUMNS}, got {template_cols}")
    required = {"origin","destination","route_type","quantity_tonnes","distance_km","freight_cost_inr","shipment_date"}
    if not required.issubset(shipments.columns):
        raise ValueError(f"Missing shipment columns: {sorted(required-set(shipments.columns))}")
    if not {"note_id","date","applies_to","note"}.issubset(notes.columns):
        raise ValueError("context_notes.csv requires note_id,date,applies_to,note")
    return shipments, notes

def weekly_metrics(shipments: pd.DataFrame) -> pd.DataFrame:
    s=shipments.copy()
    s["shipment_date"]=pd.to_datetime(s["shipment_date"], errors="raise")
    s["quantity_tonnes"]=pd.to_numeric(s["quantity_tonnes"], errors="raise")
    s["distance_km"]=pd.to_numeric(s["distance_km"], errors="raise")
    s["freight_cost_inr"]=pd.to_numeric(s["freight_cost_inr"], errors="raise")
    if (s["quantity_tonnes"]<=0).any() or (s["distance_km"]<=0).any():
        raise ValueError("Quantity and distance must be positive.")
    s["week_of"]=s["shipment_date"].dt.to_period("W-SUN").dt.start_time
    s["route"]=s["origin"].astype(str).str.strip()+"-"+s["destination"].astype(str).str.strip()
    s["tonne_km"]=s["quantity_tonnes"]*s["distance_km"]
    w=(s.groupby(["route","route_type","week_of"],as_index=False)
         .agg(total_cost=("freight_cost_inr","sum"),total_tonne_km=("tonne_km","sum")))
    w["cost_per_tonne_km"]=w["total_cost"]/w["total_tonne_km"]
    w=w.sort_values(["route","week_of"]).reset_index(drop=True)
    w["own_baseline"]=w.groupby("route")["cost_per_tonne_km"].transform(
        lambda x:x.shift(1).rolling(8,min_periods=1).mean())
    w["own_baseline_n"]=w.groupby("route").cumcount().clip(upper=8)
    peer=[]
    for r in w.itertuples(index=False):
        p=w[(w.week_of==r.week_of)&(w.route_type==r.route_type)&(w.route!=r.route)]
        peer.append(p.cost_per_tonne_km.mean() if len(p) else float("nan"))
    w["peer_baseline"]=peer
    w["own_change_pct"]=(w.cost_per_tonne_km/w.own_baseline-1)*100
    w["peer_change_pct"]=(w.cost_per_tonne_km/w.peer_baseline-1)*100
    return w

def eligible_notes(notes: pd.DataFrame, route: str, week_start: pd.Timestamp):
    """Conservative route/date matching. Multi-day incident N001 is explicit in its text."""
    week_end=week_start+pd.Timedelta(days=6)
    found=[]
    for _,n in notes.iterrows():
        applies=str(n["applies_to"]).strip()
        if applies not in (route,"All Routes"): continue
        try: nd=pd.Timestamp(str(n["date"]))
        except Exception: continue
        txt=str(n["note"]).lower()
        # Explicit multi-day incident: match any week overlapping stated date interval.
        if "from feb 24 to mar 8" in txt and applies==route:
            if week_start<=pd.Timestamp("2025-03-08") and week_end>=pd.Timestamp("2025-02-24"):
                found.append(n); continue
        # Route-specific or global note is date-local unless a continuing change is stated.
        continuing=("starting this week" in txt or "starting" in txt or "from this week" in txt)
        if week_start<=nd<=week_end or (applies=="All Routes" and continuing and nd<=week_end):
            found.append(n)
    return found

def note_is_positive_evidence(note):
    txt=str(note["note"]).lower()
    disqualifiers=["costs were not significantly affected","no major disruptions","remained normal",
                   "not part of this dataset","no significant disruptions","without a rate change"]
    return not any(x in txt for x in disqualifiers)

def choose_note(notes, route, week_start):
    c=[n for n in eligible_notes(notes,route,week_start) if note_is_positive_evidence(n)]
    # Prefer route-specific notes, then nearest date, then note id.
    c.sort(key=lambda n:(str(n["applies_to"])=="All Routes",abs((pd.Timestamp(str(n["date"]))-week_start).days),str(n["note_id"])))
    return c[0] if c else None

def fmt_change(v):
    return "N/A (no baseline)" if pd.isna(v) else f"{v:+.1f}%"

def llm_explanation(note, route, week, current, own_pct, peer_pct, model, usage):
    """Use OpenRouter only for wording; candidate note already passed local route/date checks."""
    api_key=os.getenv("OPENROUTER_API_KEY","").strip()
    if not api_key or note is None:
        return None
    system=("You write concise audit explanations for freight-cost anomaly analysis. "
            "Use only the supplied candidate note and computed facts. Do not add causes, dates, or claims not present. "
            "Return JSON with one key: reason. Do not decide the verdict or note ID.")
    payload={"model":model,"temperature":0,"messages":[
        {"role":"system","content":system},
        {"role":"user","content":json.dumps({
            "route":route,"week_of":str(pd.Timestamp(week).date()),
            "cost_per_tonne_km":round(float(current),4),
            "change_vs_own_history_pct":None if pd.isna(own_pct) else round(float(own_pct),2),
            "change_vs_similar_routes_pct":None if pd.isna(peer_pct) else round(float(peer_pct),2),
            "validated_note":{"note_id":str(note["note_id"]),"date":str(note["date"]),"applies_to":str(note["applies_to"]),"text":str(note["note"])}
        },ensure_ascii=False)}
    ]}
    try:
        import requests
        response=requests.post(API_URL,headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json",
            "HTTP-Referer":"https://localhost","X-Title":"FreightTiger Case Study"},json=payload,timeout=45)
        response.raise_for_status()
        data=response.json()
        u=data.get("usage",{})
        usage["calls"]+=1
        usage["input_tokens"]+=int(u.get("prompt_tokens",0) or 0)
        usage["output_tokens"]+=int(u.get("completion_tokens",0) or 0)
        content=data["choices"][0]["message"]["content"]
        try:
            parsed=json.loads(content)
            reason=str(parsed.get("reason","")).strip()
        except Exception:
            reason=content.strip()
        # Safety guard: reject empty or unsupported note-id substitutions.
        if not reason or str(note["note_id"]) not in reason:
            return f"Cost rise has supporting context in note {note['note_id']} dated {note['date']}: {note['note']}"
        return reason
    except Exception as exc:
        usage["errors"]+=1
        usage["last_error"]=str(exc)[:240]
        return None

def build_output(weekly, notes, model, usage):
    rows=[]
    for r in weekly.itertuples(index=False):
        own_rise=pd.notna(r.own_change_pct) and r.own_change_pct>=OWN_RISE_THRESHOLD_PCT
        peer_rise=pd.notna(r.peer_change_pct) and r.peer_change_pct>=PEER_RISE_THRESHOLD_PCT
        rising=bool(own_rise or peer_rise)
        note=choose_note(notes,r.route,r.week_of) if rising else None
        if rising and note is not None:
            flagged="No (justified)"
            note_id=str(note["note_id"])
            reason=llm_explanation(note,r.route,r.week_of,r.cost_per_tonne_km,r.own_change_pct,r.peer_change_pct,model,usage)
            if not reason:
                reason=f"Cost rise has supporting context in note {note_id} dated {note['date']}: {note['note']}"
        elif rising:
            flagged="Yes"; note_id=""
            reason="Unexplained cost rise: no note was found that supports this route and week. Flagged for human review; no cause is inferred."
        else:
            flagged="No"; note_id=""
            reason="No material rise crossed the screening thresholds; no anomaly flag raised."
        own=fmt_change(r.own_change_pct)
        if r.own_baseline_n==0: own="N/A (no prior weeks)"
        elif r.own_baseline_n<8 and pd.notna(r.own_change_pct): own+=f" vs prior {int(r.own_baseline_n)} available week(s)"
        else: own+=" vs trailing 8-week average"
        peer=fmt_change(r.peer_change_pct)
        if pd.isna(r.peer_baseline): peer="N/A (no other same-type route that week)"
        else: peer+=" vs same-type routes this week"
        rows.append({"route":r.route,"week_of":pd.Timestamp(r.week_of).strftime("%Y-%m-%d"),
          "cost_per_tonne_km":round(float(r.cost_per_tonne_km),4),"vs_own_history":own,
          "vs_similar_routes":peer,"flagged":flagged,"matched_note_id":note_id,"reason":reason})
    return pd.DataFrame(rows,columns=OUTPUT_COLUMNS)

def run(data_dir: Path, output_path: Path, model: str):
    shipments,notes=load_inputs(data_dir)
    weekly=weekly_metrics(shipments)
    usage={"calls":0,"input_tokens":0,"output_tokens":0,"errors":0,"last_error":""}
    result=build_output(weekly,notes,model,usage)
    result.to_csv(output_path,index=False)
    log=(f"Input shipment rows: {len(shipments)}\nOutput route-week rows: {len(result)}\n"
         f"Model: {model}\nLLM calls successfully completed: {usage['calls']}\n"
         f"Input tokens reported by provider: {usage['input_tokens']}\n"
         f"Output tokens reported by provider: {usage['output_tokens']}\n"
         "Model cost: check the model's current OpenRouter pricing and your provider usage dashboard; "
         "this script does not invent a rate or claim an unreported cost.\n"
         f"LLM errors/fallbacks: {usage['errors']}\n"
         f"API key configured: {'yes' if os.getenv('OPENROUTER_API_KEY','').strip() else 'no'}\n")
    if usage["last_error"]: log+="Last error: "+usage["last_error"]+"\n"
    return result,log

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--output",type=Path,default=None)
    ap.add_argument("--model",default=os.getenv("OPENROUTER_MODEL",MODEL_DEFAULT))
    ap.add_argument("--repeat-check",action="store_true",help="run three times and compare output hashes (uses LLM calls if key is set)")
    args=ap.parse_args()
    out=args.output or args.data_dir/"output.csv"
    result,log=run(args.data_dir,out,args.model)
    (args.data_dir/"token_cost_log.txt").write_text(log,encoding="utf-8")
    print(f"Processed {len(result)} route-week rows -> {out}")
    print(f"Unexplained flags: {(result.flagged=='Yes').sum()}")
    print(f"Context-supported rises: {(result.flagged=='No (justified)').sum()}")
    print(log)
    if args.repeat_check:
        hashes=[]
        for i in range(3):
            p=args.data_dir/f".repeat_{i+1}.csv"
            run(args.data_dir,p,args.model)
            hashes.append(hashlib.sha256(p.read_bytes()).hexdigest()); p.unlink()
        same=len(set(hashes))==1
        (args.data_dir/"reproducibility_check.txt").write_text(
            "Three-run byte-identical output: "+("PASS" if same else "FAIL")+"\n"+
            "\n".join(f"Run {i+1} SHA-256: {h}" for i,h in enumerate(hashes))+"\n"+
            "Note: with an LLM API key configured, wording can vary even at temperature 0; flags, metrics and note IDs are computed independently and must be compared separately.\n",encoding="utf-8")
        print("Three-run byte-identical output:", "PASS" if same else "FAIL")
if __name__=="__main__": main()
