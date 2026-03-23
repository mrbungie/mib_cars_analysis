import numpy as np
from collections import Counter

def digit_weirdness(n):
    s = str(abs(n)).replace(".", "")
    
    # normalize zeros
    s = s.lstrip("0") or "0"

    # --- last digit ---
    d = int(s[-1])
    if d == 0:
        last = -np.log(0.25)
    elif d == 5:
        last = -np.log(0.20)
    else:
        last = -np.log(0.075)

    # --- repetition ---
    max_run = 1
    current = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    repetition = max_run - 1

    # --- patterns ---
    pattern = 0
    if s == s[::-1] and len(s) > 2:
        pattern += 2
    if s in ("123456789", "987654321", "0123456789", "9876543210"):
        pattern += 3
    if len(set(s)) <= 2 and len(s) > 2:
        pattern += 2

    # --- entropy ---
    counts = Counter(s)
    probs = np.array(list(counts.values())) / len(s)
    H = -np.sum(probs * np.log(probs + 1e-12))
    entropy = (np.log(10) - H)

    # --- roundness ---
    if s.endswith("000"):
        roundness = 3
    elif s.endswith("00"):
        roundness = 2
    elif s.endswith("0"):
        roundness = 1
    else:
        roundness = 0

    # --- final score ---
    score = (
        1.5 * last
        + 1.5 * repetition
        + 2.0 * pattern
        + 1.0 * entropy
        + 2.0 * roundness
    )

    return float(score)