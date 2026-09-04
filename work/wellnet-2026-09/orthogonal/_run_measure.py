import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import adyn_same_object as S

res = json.load(open("orthogonal_results.json"))
out = {"measure": S.measure_stage(res, streams="6d+leverage")}
json.dump(out, open("_side_measure.json", "w"), indent=1)
print("MEASURE DONE")
