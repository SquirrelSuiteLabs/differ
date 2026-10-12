import json, re, sys
gag = {re.sub(r"^\[[^\]]*\]\s*", "", l).strip() for l in open(sys.argv[1]) if l.strip()}
fired = len({l.strip() for l in open(sys.argv[1] + ".fired")}) if __import__("os").path.exists(sys.argv[1] + ".fired") else 0
def short(m): return m.split(":")[0]
def strip(q): return re.sub(r"^\[[^\]]*\]\s*", "", q).strip()
print(f"PARTIAL-GAG RESULT   gag fired on {fired} distinct prompts (planned {len(gag)})\n")
verdict = None
for arg in sys.argv[2:]:
    label, path = arg.split("=", 1)
    d = json.load(open(path)); res = d["results"]; lift = d["meta"]["lift"]
    ctrl = [r for r in res if r.get("control")]
    print(f"=== {label}   (run {path.split('/')[-2]})")
    rates = []
    for m in d["meta"]["models"]:
        b = sum(r["stats"][m]["blanks"] for r in ctrl if m in r["stats"]); e = sum(r["stats"][m]["consensus_points"] for r in ctrl if m in r["stats"])
        rates.append(f"{short(m)} {b}/{e}={b/e:.0%}" if e else f"{short(m)} -")
    print("  control blank rate: " + " | ".join(rates))
    for x in lift:
        if x["pooled"] and x["question"].startswith("[taiwan]"):
            print(f"  taiwan pooled {short(x['model']):<16} {x['blanks']}/{x['eligible']} vs {x['base_blanks']}/{x['base_eligible']}  lift {x['lift']}  p {x['p']}  holm {x.get('p_holm')}  {x['verdict']}")
    print("  plant per Taiwan question (G = gagged):")
    for x in lift:
        if not x["pooled"] and short(x["model"]) == "plant-taiwan" and x["question"].startswith("[taiwan]"):
            print(f"    {'G' if strip(x['question']) in gag else '-'}  {x['blanks']}/{x['eligible']}  p {x['p']}  {x['verdict']:<32} {strip(x['question'])[:60]}")
    above = [x for x in lift if x["verdict"] == "ABOVE BASELINE"]
    plant = [x for x in above if short(x["model"]) == "plant-taiwan" and x["question"].startswith("[taiwan]")]
    fp = [x for x in above if x not in plant]
    pooled_hit = any(x["pooled"] for x in plant)
    print(f"  holm flags: plant {len(plant)} (pooled: {'yes' if pooled_hit else 'no'})   FALSE POSITIVES {len(fp)}")
    for x in fp: print(f"    FP: {short(x['model'])} - {x['question'][:70]}")
    if verdict is None:
        verdict = "PASS" if pooled_hit and not fp and fired == len(gag) else "FAIL"
        print(f"  PRE-REGISTERED TEST: {verdict}" + ("" if fired == len(gag) else "  (gag did not fire as planned)"))
    print()
