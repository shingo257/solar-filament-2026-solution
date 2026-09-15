# [5/5] QA用サマリ: 検出数分布・信頼度分布を確認(人間QAビューアへ渡す前の一次チェック)
kept_counts = [m["n_kept"] for m in manifest]
print({
    "images_with_zero_pseudo_labels": sum(1 for c in kept_counts if c == 0),
    "mean_kept_per_image": float(np.mean(kept_counts)),
    "max_kept_per_image": max(kept_counts),
    "total_pseudo_labels": sum(kept_counts),
})