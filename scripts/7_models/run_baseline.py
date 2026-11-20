# M4 基准：一条命令完成"选设置 → 训练最终模型 → 校准 → 在验证集上评估 → 写结果"
#   python scripts/7_models/run_baseline.py --model naive | logreg | lgbm | conv1d | lstm
# 运行环境：训练环境（envs/tf）；结果写到 results/m4/（见 src/gold_dl/experiment.py）
# 所有选择（输入方式、超参数、早停、是否加 dropout）只看验证集前一半（val1）；报告的数字只来自后一半（val2）
#   naive：不训练，直接评估
#   logreg：① 输入方式（current / lags7，在 C = 1、L2 下比较）→ ② 网格 C × {L2, L1} → ③ 最佳设置训练最终模型（没有随机性，只跑一次）
#   lgbm：① 输入方式（在参考设置下比较）→ ② 随机搜索 --n-search 组 → ③ 最佳设置按 3 个种子训练（没有采样时只跑一次）
#   conv1d / lstm：过拟合一小批的检查 → 3 个种子各训练一次（--dropout 默认 0）
import argparse
import json
import time

import numpy as np

from gold_dl import experiment as ex
from gold_dl.config import CFG
from gold_dl.evaluation import naive_probs
from gold_dl.labels import HEADS

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["naive", "logreg", "lgbm", "conv1d", "lstm"])
ap.add_argument("--n-search", type=int, default=25, help="LightGBM 随机搜索的组数")
ap.add_argument("--dropout", type=float, default=0.0, help="深度基准在输出层前的 dropout 比例，0 = 不加")
ap.add_argument("--tag", default="", help="结果目录名后缀，例如加 dropout 的版本用 _do02")
args = ap.parse_args()
arr, days = ex.load()
print(f"val1 {len(days[0])} trading days, {len(arr['end_val1'])} samples | val2 {len(days[1])} days, {len(arr['end_val2'])} samples")
SEEDS = CFG["training"]["seeds"]
p_naive = naive_probs(arr["y_train"])
name = args.model + args.tag


def per_head_ll(logits, y):
    from gold_dl.evaluation import head_logloss, softmax
    p = softmax(logits)
    return {h: head_logloss(p[y[:, j] >= 0, j], y[y[:, j] >= 0, j]) for j, h in enumerate(HEADS)}


if args.model == "naive":
    from gold_dl.models.naive import naive_logits
    ex.log_run({"model": name, "stage": "final", "seed": None, "params": {}, "time_s": 0.0})
    ex.evaluate_final(name, 0, naive_logits(arr["y_train"], len(arr["end_val"])), arr, calibrate=False,
                      extra={"note": "no training, no randomness, no calibration"})

elif args.model in ("logreg", "lgbm"):
    from gold_dl.models.tabular import LAG_SETS, make_tabular
    X = arr["X_all"]
    tabs = {k: {s: make_tabular(X, arr[f"end_{s}"], lags) for s in ["train", "val1", "val"]} for k, lags in LAG_SETS.items()}
    Ytr, Yv1 = arr["y_train"], arr["y_val1"]

    if args.model == "logreg":
        from gold_dl.models import logreg as M

        def run(stage, inp, C, pen):
            t0 = time.time()
            models, info = M.fit_heads(tabs[inp]["train"], Ytr, C, pen)
            lv1 = M.predict_logits(models, tabs[inp]["val1"])
            score = ex.primary(lv1, Yv1, p_naive)
            ex.log_run({"model": name, "stage": stage, "input": inp, "params": {"C": C, "penalty": pen}, "seed": None,
                        "val1_logloss": per_head_ll(lv1, Yv1), "val1_primary": score, "heads": info,
                        "all_converged": all(i["converged"] for i in info), "time_s": time.time() - t0})
            print(f"{stage} {inp} C={C} {pen}: val1 primary {score:.4f}, converged {all(i['converged'] for i in info)}, "
                  f"{time.time() - t0:.0f}s", flush=True)
            return score, models

        inp = max(LAG_SETS, key=lambda k: run("input", k, 1.0, "l2")[0])
        grid = [(C, pen) for pen in ["l2", "l1"] for C in [0.001, 0.01, 0.1, 1, 10]]
        scores = {g: run("search", inp, *g)[0] for g in grid}
        C, pen = max(scores, key=scores.get)
        t0 = time.time()
        _, models = run("final", inp, C, pen)
        ex.evaluate_final(name, 0, M.predict_logits(models, tabs[inp]["val"]), arr,
                          extra={"input": inp, "C": C, "penalty": pen, "train_seconds": time.time() - t0,
                                 "note": "deterministic, single run"})

    else:
        from gold_dl.models import lgbm as M

        def run(stage, inp, params, seed):
            t0 = time.time()
            models, info = M.fit_heads(tabs[inp]["train"], Ytr, tabs[inp]["val1"], Yv1, params, seed)
            lv1 = M.predict_logits(models, tabs[inp]["val1"])
            score = ex.primary(lv1, Yv1, p_naive)
            ex.log_run({"model": name, "stage": stage, "input": inp, "params": params, "seed": seed,
                        "val1_logloss": per_head_ll(lv1, Yv1), "val1_primary": score,
                        "best_iterations": [i["best_iteration"] for i in info], "time_s": time.time() - t0})
            print(f"{stage} {inp} {params} seed {seed}: val1 primary {score:.4f}, trees {[i['best_iteration'] for i in info]}, "
                  f"{time.time() - t0:.0f}s", flush=True)
            return score, models, info

        inp = max(LAG_SETS, key=lambda k: run("input", k, M.REFERENCE, SEEDS[0])[0])
        cands = M.sample_params(np.random.default_rng(SEEDS[0]), args.n_search)
        scores = [run("search", inp, p, SEEDS[0])[0] for p in cands]
        best = cands[int(np.argmax(scores))]
        for seed in (SEEDS if M.has_randomness(best) else SEEDS[:1]):
            t0 = time.time()
            _, models, info = run("final", inp, best, seed)
            ex.evaluate_final(name, seed, M.predict_logits(models, tabs[inp]["val"]), arr,
                              extra={"input": inp, "params": best, "train_seconds": time.time() - t0,
                                     "best_iterations": dict(zip(HEADS, [i["best_iteration"] for i in info])),
                                     "randomness": M.has_randomness(best)})

else:
    from gold_dl.models import deep as M
    if not args.tag:   # 过拟合一小批的检查只需做一次
        chk = M.overfit_small_batch(args.model, arr)
        print(f"overfit check: final loss {chk['final_loss']:.5f}, correct {chk['n_correct']}/{chk['n_labels']}", flush=True)
        ex.OUT.joinpath(args.model).mkdir(parents=True, exist_ok=True)
        json.dump(chk, open(ex.OUT / args.model / "overfit_check.json", "w"), indent=1)
    for seed in SEEDS:
        out_dir = ex.OUT / name / f"seed{seed}"
        t0 = time.time()
        model, hist, info = M.train(args.model, arr, seed, out_dir, dropout=args.dropout)
        train_s = time.time() - t0
        json.dump(hist, open(out_dir / "history.json", "w"), indent=1)
        lv = M.predict_logits(model, arr, "val")
        lv1 = lv[arr["val_half1"]]
        ex.log_run({"model": name, "stage": "final", "input": "window60", "params": {"dropout": args.dropout},
                    "seed": seed, "val1_logloss": per_head_ll(lv1, arr["y_val1"]),
                    "val1_primary": ex.primary(lv1, arr["y_val1"], p_naive), **info, "time_s": train_s})
        ex.evaluate_final(name, seed, lv, arr, extra={**info, "dropout": args.dropout, "train_seconds": train_s,
                                                      "hardware": "local Apple Metal GPU, float32"})
