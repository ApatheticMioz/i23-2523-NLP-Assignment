import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

checks = []


def add_check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


# Core artifact checks
required_files = [
    "embeddings/tfidf_matrix.npy",
    "embeddings/ppmi_matrix.npy",
    "embeddings/embeddings_w2v.npy",
    "embeddings/word2idx.json",
    "embeddings/part1_report.json",
    "data/pos_train.conll",
    "data/pos_val.conll",
    "data/pos_test.conll",
    "data/ner_train.conll",
    "data/ner_val.conll",
    "data/ner_test.conll",
    "data/part2_report.json",
    "data/part3_report.json",
    "data/part3_bilstm_vs_transformer.txt",
    "models/bilstm_pos.pt",
    "models/bilstm_ner.pt",
    "models/transformer_cls.pt",
]
for rel in required_files:
    add_check(f"file exists: {rel}", (ROOT / rel).exists())

part1 = json.loads((ROOT / "embeddings/part1_report.json").read_text(encoding="utf-8"))
part2 = json.loads((ROOT / "data/part2_report.json").read_text(encoding="utf-8"))
part3 = json.loads((ROOT / "data/part3_report.json").read_text(encoding="utf-8"))

# Part 1
add_check("part1 tfidf shape", part1.get("tfidf_shape") == [240, 10001], str(part1.get("tfidf_shape")))
add_check("part1 ppmi shape", part1.get("ppmi_shape") == [10001, 10001], str(part1.get("ppmi_shape")))
add_check("part1 analogy count", len(part1.get("analogy_results", [])) == 10, str(len(part1.get("analogy_results", []))))
add_check("part1 analogy top3 size", all(len(x.get("top3", [])) == 3 for x in part1.get("analogy_results", [])))
add_check("part1 analogy >=5 correct", part1.get("analogy_correct_count", 0) >= 5, str(part1.get("analogy_correct_count", 0)))
add_check("part1 4 condition mrr", len(part1.get("condition_mrr", {})) == 4)
req_nbrs = part1.get("required_neighbors_c3", {})
add_check("part1 8 required query words", len(req_nbrs) == 8, str(len(req_nbrs)))
add_check(
    "part1 required neighbors top10",
    all(len(v.get("neighbors", [])) == 10 for v in req_nbrs.values()),
)

# Part 2
add_check("part2 500 sentences", part2.get("sentence_count") == 500, str(part2.get("sentence_count")))
add_check("part2 split 70/15/15", part2.get("split_sizes") == {"train": 350, "val": 75, "test": 75}, str(part2.get("split_sizes")))
lex = part2.get("lexicon_sizes", {})
add_check("part2 noun lexicon >=200", lex.get("noun", 0) >= 200, str(lex.get("noun", 0)))
add_check("part2 verb lexicon >=200", lex.get("verb", 0) >= 200, str(lex.get("verb", 0)))
add_check("part2 adj lexicon >=200", lex.get("adj", 0) >= 200, str(lex.get("adj", 0)))
add_check("part2 per gazetteer >=50", lex.get("per_gazetteer", 0) >= 50, str(lex.get("per_gazetteer", 0)))
add_check("part2 loc gazetteer >=50", lex.get("loc_gazetteer", 0) >= 50, str(lex.get("loc_gazetteer", 0)))
add_check("part2 org gazetteer >=30", lex.get("org_gazetteer", 0) >= 30, str(lex.get("org_gazetteer", 0)))

pos_res = part2.get("pos_results", {})
add_check("part2 pos has frozen+finetuned val f1", "frozen_val_f1" in pos_res and "finetuned_val_f1" in pos_res)
add_check("part2 pos has test metrics", "test_accuracy" in pos_res and "test_macro_f1" in pos_res)
confused = pos_res.get("top_confused_pairs", [])
add_check("part2 top 3 confused pairs", len(confused) == 3, str(len(confused)))
add_check(
    "part2 confused pairs have >=2 examples",
    all(len(x.get("examples", [])) >= 2 for x in confused),
)

ner_res = part2.get("ner_results", {})
add_check("part2 ner crf metric present", "test_overall_f1_crf" in ner_res)
add_check("part2 ner softmax metric present", "test_overall_f1_softmax" in ner_res)
crf_report = ner_res.get("entity_report_crf", {})
soft_report = ner_res.get("entity_report_softmax", {})
for key in ["PER", "LOC", "ORG", "MISC", "micro avg", "macro avg"]:
    add_check(f"part2 crf report has {key}", key in crf_report)
    add_check(f"part2 softmax report has {key}", key in soft_report)
add_check("part2 ner false positives >=5", len(ner_res.get("false_positives", [])) >= 5, str(len(ner_res.get("false_positives", []))))
add_check("part2 ner false negatives >=5", len(ner_res.get("false_negatives", [])) >= 5, str(len(ner_res.get("false_negatives", []))))
abl = part2.get("ablations", {})
for k in [
    "A1_unidirectional_lstm",
    "A2_no_dropout",
    "A3_random_embeddings",
    "A4_softmax_instead_of_crf",
]:
    add_check(f"part2 ablation has {k}", k in abl)

# Part 3
for split_name in ["train", "val", "test"]:
    dist = part3.get("class_distribution", {}).get(split_name, {})
    add_check(f"part3 {split_name} has 5 classes", len(dist) == 5, str(dist))
    add_check(f"part3 {split_name} all classes nonzero", all(v > 0 for v in dist.values()), str(dist))

tr = part3.get("transformer", {})
add_check("part3 transformer test_acc present", "test_acc" in tr)
add_check("part3 transformer macro_f1 present", "test_macro_f1" in tr)
add_check("part3 transformer best epoch present", "best_epoch" in tr)
add_check("part3 transformer has epoch times", len(tr.get("epoch_times", [])) == 20, str(len(tr.get("epoch_times", []))))

bl = part3.get("bilstm_baseline", {})
add_check("part3 bilstm test_acc present", "test_acc" in bl)
add_check("part3 bilstm macro_f1 present", "test_macro_f1" in bl)

heatmaps = part3.get("attention_heatmaps", [])
add_check("part3 has >=6 attention heatmaps", len(heatmaps) >= 6, str(len(heatmaps)))

# Print summary
passed = sum(1 for _, ok, _ in checks if ok)
total = len(checks)
print(f"PASS {passed}/{total}")
for name, ok, detail in checks:
    status = "OK" if ok else "FAIL"
    if detail:
        print(f"[{status}] {name} :: {detail}")
    else:
        print(f"[{status}] {name}")
