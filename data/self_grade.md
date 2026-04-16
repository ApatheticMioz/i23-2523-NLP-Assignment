# Self-Grade (Rubric Coverage)

## Summary

- Targeted total: 75/75
- Estimated by artifact and checklist coverage: 75/75
- GitHub component: 5/5

## Part 1 (25/25)

- Term-document + vocab cap: done (`scripts/part1_embeddings.py`, `embeddings/word2idx.json`)
- TF-IDF formula + save: done (`embeddings/tfidf_matrix.npy`)
- Topic-wise discriminative terms: done (`embeddings/part1_report.json`)
- Co-occurrence + PPMI + save: done (`embeddings/ppmi_matrix.npy`)
- t-SNE + neighbors: done (`figures/part1_tsne_top200.png`, report neighbors)
- Skip-gram from scratch + BCE + V/U: done (`scripts/part1_embeddings.py`)
- Loss curve + embedding save: done (`figures/part1_skipgram_loss_c3.png`, `embeddings/embeddings_w2v.npy`)
- Analogy + required query evaluations: done (`embeddings/part1_report.json`)
- C1-C4 + MRR: done (`embeddings/part1_report.json`)

## Part 2 (25/25)

- 500-sentence stratified sampling: done (`data/part2_report.json`)
- POS rule tagger + lexicons: done (lexicon sizes in report)
- NER BIO + gazetteers: done (gazetteer sizes in report)
- 70/15/15 split + label distributions: done (report + CoNLL files)
- 2-layer BiLSTM + dropout + masking: done (`scripts/part2_sequence_labeling.py`)
- CRF + Viterbi implementation: done (same script)
- Frozen vs fine-tuned comparison: done (POS and NER metrics in report)
- POS metrics + confusion + confused pairs: done (report + figure)
- NER metrics + CRF vs softmax + FP/FN: done (report)
- Ablations A1-A4: done (report)

## Part 3 (20/20)

- 5-class dataset + stratified split: done (`data/part3_report.json`)
- Custom attention modules: done (`scripts/part3_transformer_classifier.py`)
- Multi-head, FFN, sinusoidal PE, Pre-LN x4: done (same script)
- CLS classifier + AdamW + cosine warmup: done (same script)
- Curves + checkpoint: done (`figures/transformer_loss_curve.png`, `models/transformer_cls.pt`)
- Test accuracy + macro-F1 + confusion matrix: done (`data/part3_report.json`, figure)
- Attention heatmaps (>=2 heads, 3 samples): done (`figures/part3_attention_sample*_head*.png`)
- BiLSTM vs Transformer (5 required prompts): done (`data/part3_bilstm_vs_transformer.txt`)

## GitHub Component (5/5)

- Public repository with required naming: done
- All code committed: done
- Meaningful incremental commits >= 5: done
- README reproduction instructions: done (`README.md`)

## Repository

https://github.com/ApatheticMioz/i23-2523-NLP-Assignment