import glob
import re

files = glob.glob('ml_bot/*.py')
files.append('update_traders.py')

for fpath in files:
    with open(fpath, 'r', encoding='utf-8') as f:
        code = f.read()
        
    # Replace feature list
    old_feat = "'dxy', 'us10y', 'atr_14'"
    new_feat = "'dxy', 'us10y', 'atr_14', 'day_of_week'"
    code = code.replace(old_feat, new_feat)
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(code)
        
print("Updated features list in all python files.")
