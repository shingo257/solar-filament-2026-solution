# scratch/ インデックス — 2026-11-01 時点の分類

197 の .py スクリプト（+ viewer 用 .html・`_survey*/` 一時物）を分類整理。
**移動はしない** — `RESEARCH_LOG.md` が
ファイルパスを provenance として引用し、tests/ も `from scratch.X` を直接
import するため、経路は契約として凍結する。新規スクリプトを追加したら
本表に追記すること。

## A. 稼働 tooling（変更時はテスト・再生成確認が必須）

### A-1. テストで直接 import されるモジュール (22)

tests/ が `from scratch.X` で利用。シグネチャ変更は対応テストも要更新。

- `analyze_s0_blinded_review.py`
- `audit_s0_blinded_review_readiness.py`
- `build_gong_ssl_corpus.py`
- `build_s1_full_development_set_context_features.py`
- `build_verifier_dataset.py`
- `calibrate_s1_endpoint_margins.py`
- `evaluate_s1_conditional_rescue_headroom.py`
- `evaluate_s1_prediction_geometry_gate.py`
- `pretrain_gong_ssl.py`
- `pretrain_unet_encoder_ssl.py`
- `rerun_v95_gate_corrected_labels.py`
- `run_s0_audit.py`
- `run_s0_baseline_rescore.py`
- `run_s0_blinded_review.py`
- `run_s1_proposal_marginal_utility_pilot.py`
- `run_v42_verifier_official_eval.py`
- `run_v80_hybrid_candidates.py`
- `run_v80_scorer_eval.py`
- `run_verifier_filter_eval.py`
- `train_refiner_fold.py`
- `train_verifier.py`
- `v73_split_provenance.py`

### A-2. 分割 companion（A-1 の _impl/_main/_cells） (8)

500行ゴール分割で分離した実装体。wrapper 側の import で解決される。

- `evaluate_s1_conditional_rescue_headroom_impl.py`
- `run_s0_audit_impl.py`
- `run_s0_baseline_rescore_impl.py`
- `run_s0_baseline_rescore_main.py`
- `run_s0_blinded_review_impl.py`
- `run_s1_proposal_marginal_utility_pilot_impl.py`
- `run_v80_scorer_eval_impl.py`
- `run_v80_scorer_eval_main.py`

### A-3. 提出物生成・重み export (5)

最終提出パイプラインの構成要素。byte-identical 再現検証が契約。

- `build_prod_repro_colab.py`
- `build_submission_reproduction.py`
- `compare_submissions.py`
- `export_submission_weights.py`
- `run_test_submission.py`

### A-4. 評価・分析 tooling (7)

RESEARCH_LOG §2.x の数値を生成した解析スクリプト（実験系以外の現役）。

- `analyze_error_decomposition.py`
- `analyze_error_decomposition_impl.py`
- `analyze_nearmiss_area_bias.py`
- `analyze_s1_harm_target_feasibility.py`
- `register_s1_decision_policy.py`
- `validate_s1_harm_first_model_spec.py`
- `validate_v35_submission.py`


## B. 凍結実験（provenance のみ・編集しない）

### B-1. self-hash builder（分割・改名 禁止） (12)

`PROTOCOL_SHA256 = sha256(__file__)` の自己ハッシュを持ち、凍結 `expected_contract` artifact と紐付く。**REFACTOR_PLAN Step 8 免除確定**。

- `build_v57_colab_dense_oversample_pilot.py`
- `build_v58_conf_sweep_probe.py`
- `build_v59_boundary_dt_pilot.py`
- `build_v60_copypaste_pilot.py`
- `build_v61_sam2_refiner_pilot.py`
- `build_v62_error_anatomy.py`
- `build_v64_selector_pilot.py`
- `build_v65_tta_consensus.py`
- `build_v66_rfdetr_union.py`
- `build_v87_yolo_union_label.py`
- `build_v93_consensus_class.py`
- `fetch_temporal_neighbors.py`

### B-2. 実験 one-off（v19〜v93 系） (100)

採否判定済み実験の生成器・実行器。RESEARCH_LOG §2.x が結果を引用。

- `_v48_cell1.py`
- `_v48_cell2.py`
- `_v48_cell3.py`
- `_v48_cell4.py`
- `_v48_cell5.py`
- `_v49_cell1.py`
- `_v49_cell2.py`
- `_v49_cell3.py`
- `_v49_cell4.py`
- `_v49_cell5.py`
- `_v49_cell6.py`
- `_v49_cell7.py`
- `_v49_cell8.py`
- `build_v24_uploaded_zip_notebook.py`
- `build_v25_uploaded_zip_notebook.py`
- `build_v26_realdata_yolo_refiner.py`
- `build_v27_offline_mediseg.py`
- `build_v28_offline_mediseg_stable.py`
- `build_v29_yolo_mediseg_fusion.py`
- `build_v30_disjoint_masks.py`
- `build_v30_japanese_ram_adaptive.py`
- `build_v30_prize_colab.py`
- `build_v30_prize_colab_cells.py`
- `build_v30_prize_colab_final.py`
- `build_v30_standalone_repair.py`
- `build_v31_heldout_postprocess_ablation.py`
- `build_v31_heldout_postprocess_ablation_cells.py`
- `build_v32_overlap_cluster_notebook.py`
- `build_v33_public069_reproduction.py`
- `build_v33_public069_reproduction_cells.py`
- `build_v34_semantic_instance_hybrid.py`
- `build_v34_semantic_instance_hybrid_cells.py`
- `build_v35_final_submission.py`
- `build_v35_metric_realignment.py`
- `build_v35_metric_realignment_cells.py`
- `build_v35_refiner_folds12.py`
- `build_v36_infer_imgsz_probe.py`
- `build_v36_yolo_imgsz_probe.py`
- `build_v36_yolo_imgsz_probe_cells.py`
- `build_v37_kaggle_timing_probe.py`
- `build_v38_kaggle_augmentation_probe.py`
- `build_v38_kaggle_augmentation_probe_cells.py`
- `build_v39_kaggle_modelsize_probe.py`
- `build_v39_kaggle_modelsize_probe_cells.py`
- `build_v40_kaggle_augmentation_v2.py`
- `build_v40_kaggle_augmentation_v2_cells.py`
- `build_v41_colab_modelsize_v2.py`
- `build_v41_colab_modelsize_v2_cells.py`
- `build_v41_kaggle_modelsize_v2.py`
- `build_v41_kaggle_modelsize_v2_cells.py`
- `build_v42_kaggle_multiseed_probe.py`
- `build_v42_kaggle_multiseed_probe_cells.py`
- `build_v43_colab_tta_wbf_probe.py`
- `build_v44_kaggle_rfdetr_probe.py`
- `build_v45_colab_rfdetr_hires_probe.py`
- `build_v45_kaggle_rfdetr_hires_probe.py`
- `build_v46_colab_refiner_free_retrain.py`
- `build_v47_colab_refiner_scratch_retrain.py`
- `build_v48_colab_pseudo_label_generation.py`
- `build_v49_colab_pseudo_label_fold0_retrain.py`
- `build_v49_colab_pseudo_label_fold0_retrain_cells.py`
- `build_v50_no_ensemble_submission.py`
- `build_v51_edgeattnet_refiner.py`
- `build_v51_edgeattnet_refiner_cells.py`
- `build_v52_edgeattnet_submission.py`
- `build_v53_edgeattnet_ensemble_submission.py`
- `build_v54_colab_ema_pilot.py`
- `build_v54_colab_ema_pilot_cells.py`
- `build_v55_colab_multichannel_refiner_pilot.py`
- `build_v55_colab_multichannel_refiner_pilot_cells.py`
- `build_v56_colab_gap_closure_probe.py`
- `build_v67_temporal_aug.py`
- `build_v73_ssl_refiner_kernel.py`
- `build_v74_ssl_confirm_colab.py`
- `build_v75_tta_guard_colab.py`
- `build_v76_ensemble_colab.py`
- `build_v80_candidates_colab.py`
- `build_v82_soft_target_refiner.py`
- `build_v82_soft_target_refiner_cells.py`
- `build_v88_yolo_cls15.py`
- `build_v88_yolo_cls15_cells.py`
- `build_v88_yolo_cls15_kaggle.py`
- `build_v89_cls15_fold01.py`
- `build_v89_cls15_fold01_cells.py`
- `build_v90_semprop.py`
- `build_v90_semprop_cells.py`
- `build_v91_semsweep.py`
- `build_v91_semsweep_cells.py`
- `build_v92_cellpose_flow.py`
- `build_v92_cellpose_flow_cells.py`
- `diagnose_v80_scorer.py`
- `run_v38_v39_raw_pq.py`
- `run_v82_aggregate.py`
- `run_v82_refiner_eval.py`
- `run_v94_v95_emission.py`
- `upgrade_gong_corpus_v2.py`
- `v68_magnetogram_probe.py`
- `v68_magnetogram_probe_impl.py`
- `v82_manifest_from_printed.py`
- `v82_manifest_repair_cell.py`

### B-3. 非バージョン build_/run_/train_ 系（凍結） (37)

名前にバージョンを持たない生成器・実行器。いずれも判定済みで凍結。

- `build_gong_ssl_kernel.py`
- `build_master_pipeline_notebook.py`
- `build_matched_param_ablation.py`
- `build_matched_param_ablation_cells.py`
- `build_near_miss_viewer_data.py`
- `build_pseudo_label_viewer_data.py`
- `build_qualitative_analysis.py`
- `build_refiner_free_retrain_manifest.py`
- `build_showcase_data.py`
- `fetch_gong_pseudo_label_pilot.py`
- `run_annotator_ceiling.py`
- `run_blob_split_probe.py`
- `run_case_diagnostics.py`
- `run_fold_oof_eval.py`
- `run_fold_oof_eval_tta.py`
- `run_min_area_sweep.py`
- `run_multiseed_ensemble_probe.py`
- `run_multiseed_wbf_probe.py`
- `run_near_miss_modes.py`
- `run_ownership_ablation.py`
- `run_p2_3_infer_local.py`
- `run_p2_3_raw.py`
- `run_p2_imgsz_oof.py`
- `run_p2_imgsz_raw_pq.py`
- `run_pipeline_stack.py`
- `run_pq_oof_rescore.py`
- `run_refiner_free_dropcontained_probe.py`
- `run_refiner_free_variant_allfold_probe.py`
- `run_refiner_free_variant_probe.py`
- `run_refiner_hybrid_variant_probe.py`
- `run_refiner_pq_ablation.py`
- `run_refiner_pq_allfold.py`
- `run_tile_infer_probe.py`
- `run_tile_pipeline_oof.py`
- `run_wbf_dropcontained_threshold_probe.py`
- `run_wbf_yolo_rfdetr_ensemble.py`
- `train_a1_scorer.py`


## C. その他

### その他ユーティリティ (5)

データ変換・viewer 用の補助。使用頻度は低いが参照記録あり。

- `convert_gong_fits_to_jpeg.py`
- `gen_nb.py`
- `preprocess_pseudo_label_images.py`
- `recount_rescue_prod_conf.py`
- `render_report_segmentation_example.py`


---

凡例: 計 196 ファイル（`__init__.py` 除く）。
分類は 2026-11-01 の機械検出（tests インポート走査 + `PROTOCOL_SHA256` grep +
命名規則）に基づく。誤分類を見つけたら該当行だけ直すこと。
