"""Rerun all new mechanics tests without rewriting their frozen receipts."""
import importlib.util,io,json,sys,unittest
from pathlib import Path
root=Path(__file__).resolve().parents[4];package=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts'))
suite=unittest.TestSuite()
files=[root/'tests/test_mond_atlas_clock_relay.py',root/'tests/test_mond_atlas_clock_scale_repair.py',root/'tests/test_mond_atlas_clock_core_repair.py',package/'physics/test_clock_mechanics.py']
for i,path in enumerate(files):
    spec=importlib.util.spec_from_file_location('clock_review_'+str(i),path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(package/'interpretation/combined-tests.txt').write_text(stream.getvalue(),encoding='utf-8')
receipt=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful())
(package/'interpretation/combined-tests.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt));assert result.wasSuccessful()
