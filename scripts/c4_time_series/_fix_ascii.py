"""Strip all non-ASCII characters from pipeline scripts so they print cleanly on Windows cp1252 terminals."""
import os

scripts = [
    'scripts/c4_time_series/c4_04_ml_preprocessing.py',
    'scripts/c4_time_series/c4_05_baseline_models.py',
    'scripts/c4_time_series/c4_06_deep_learning_models.py',
    'scripts/c4_time_series/c4_07_evaluation_report.py',
    'scripts/c4_time_series/c4_08_overfit_check.py',
]

# Map known special chars to plain ASCII equivalents
REPLACEMENTS = {
    # Box drawing
    '\u2500': '-', '\u2501': '-', '\u2502': '|', '\u254c': '-',
    '\u2550': '=', '\u2551': '|', '\u256c': '+',
    # Symbols
    '\u2705': '[OK]', '\u274c': '[ERR]', '\u26a0\ufe0f': '[WARN]',
    '\u26a0': '[WARN]', '\u2192': '->', '\u2190': '<-',
    '\u23f9': '[STOP]', '\u2714': '[OK]', '\u2717': '[X]',
    '\ud83c\udfc6': '[BEST]', '\ud83d\udfe2': '[G]',
    '\ud83d\udfe1': '[Y]', '\ud83d\udd34': '[R]',
    # Punctuation
    '\u2019': "'", '\u201c': '"', '\u201d': '"',
    '\u2013': '-', '\u2014': '--', '\u00b2': '2',
    '\u00b1': '+-',
}

for path in scripts:
    if not os.path.exists(path):
        print(f'skip (not found): {path}')
        continue
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    new = content
    for uni, asc in REPLACEMENTS.items():
        new = new.replace(uni, asc)
    # Final sweep: replace any remaining non-ASCII with '?'
    new = new.encode('ascii', errors='replace').decode('ascii')
    if new != content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new)
        print(f'fixed: {path}')
    else:
        print(f'clean: {path}')

print('All done.')
