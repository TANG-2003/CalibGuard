
from datasets import load_dataset

def normalize(prompt):
    return " ".join(str(prompt).split()).casefold()

redbench = load_dataset(
    "knoveleng/redbench",
    "ORBench",
    split="train",
)

official = load_dataset(
    "bench-llm/or-bench",
    "or-bench-hard-1k",
    split="train",
)

red_prompts = {
    normalize(row["prompt"])
    for row in redbench
}

official_prompts = {
    normalize(row["prompt"])
    for row in official
}

overlap = red_prompts & official_prompts

print("\nOR-Bench Cross-Source Overlap Audit")
print("=" * 55)

print("RedBench ORBench:", len(red_prompts))
print("Official OR-Bench Hard:", len(official_prompts))
print("Overlapping prompts:", len(overlap))

print(
    "Only in RedBench:",
    len(red_prompts - official_prompts)
)

print(
    "Only in official Hard:",
    len(official_prompts - red_prompts)
)

print(
    "Exact set match:",
    red_prompts == official_prompts
)

print("\nAudit complete.")
