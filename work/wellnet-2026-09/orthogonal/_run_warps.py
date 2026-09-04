import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warp_pool
json.dump(warp_pool.run(verbose=False), open("_warps_out.json", "w"), indent=1)
print("WARPS DONE")
