from pathlib import Path
import ast, re
root=Path(r'C:\Users\William Dennis\projects\nexus')
py_files=[p for p in root.rglob('*.py') if 'venv' not in p.parts and '__pycache__' not in p.parts]
print('PY_FILES', len(py_files))
syntax_errors=[]
for p in py_files:
    try:
        ast.parse(p.read_text(encoding='utf-8'))
    except Exception as e:
        syntax_errors.append((p, e))
if syntax_errors:
    print('SYNTAX_ERRORS')
    for p,e in syntax_errors:
        print(p, e)
patterns={
    'eval_exec': re.compile(r'\b(eval|exec)\s*\('),
    'pickle_load': re.compile(r'pickle\.load|joblib\.load'),
    'yaml_load': re.compile(r'yaml\.load\s*\('),
    'subprocess': re.compile(r'\bsubprocess\.'),
    'os_system': re.compile(r'os\.system\s*\('),
    'hardcoded_key': re.compile(r"(api[_-]?key|secret|token|password|passwd)\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    'requests_get': re.compile(r'requests\.get\s*\('),
    'requests_post': re.compile(r'requests\.post\s*\('),
    'open': re.compile(r'\bopen\s*\('),
}

hits=[]
for p in py_files:
    text=p.read_text(encoding='utf-8', errors='ignore')
    for name,pat in patterns.items():
        for m in pat.finditer(text):
            line=text.count('\n',0,m.start())+1
            hits.append((name,str(p),line,m.group(0)))
print('HITS', len(hits))
for h in hits[:200]:
    print(h)
