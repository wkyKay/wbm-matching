"""Render documentation tables from the single, adjacent results catalog; no evaluation."""

import argparse
import csv
import hashlib
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT.parent / "wbm-region-retrieval" / "results_catalog"
DOCUMENT = ROOT / "docs" / "对比方法实验结果.md"
BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"
METHODS = ("nakazawa", "wbbs", "yu", "wapirl", "wu", "wmhd")
NAMES = dict(zip(METHODS, ("Nakazawa", "WBBS", "Yu", "WaPIRL", "Wu", "WMHD")))
KS = (1, 5, 10, 20, 50, 100)
EXACT = ("Hit", "Precision", "Recall", "mAP")
COVERAGE = ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit",
            "FullCoveragePrecision", "FullCoverageRecall", "FullCoverage_mAP")
GROUPS = {
    "all": "总体query平均", "class_macro": "完整组合类别宏平均",
    "single_label": "单缺陷", "mixed_label": "混合缺陷",
    **{f"by_label_count/{n}": f"标签数{n}" for n in range(1, 5)},
}
SCOPES = (("普通test", "test"), ("跨组合test", "cross"))


def key(method, scope):
    return f"cross_{method}" if scope == "cross" else f"{method}_{scope}"


def read_csv(name):
    with (CATALOG / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "|" + "---|" * len(headers),
            *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""]


def render():
    wanted = {key(m, s) for _, s in SCOPES for m in METHODS}
    registry = {r["key"]: r for r in read_csv("registry.csv") if r["key"] in wanted}
    if set(registry) != wanted:
        raise ValueError("Missing registered evaluations")
    data = {}
    for r in read_csv("metrics.csv"):
        if r["key"] not in wanted or int(r["K"]) not in KS:
            continue
        reg = registry[r["key"]]
        for field in ("run_id", "method", "split", "protocol"):
            if r[field] != reg[field]:
                raise ValueError(f"Identity mismatch: {r['key']} {field}")
        metric = r["metric"]
        family = "exact_v1" if metric in EXACT else "coverage_v1"
        if metric not in EXACT + COVERAGE or r["metric_version"] != family:
            raise ValueError("Unexpected metric or family")
        identity = (r["key"], r["group"], metric, int(r["K"]))
        if identity in data:
            raise ValueError(f"Duplicate metric: {identity}")
        value = Decimal(r["value"]) if r["value"] else None
        if value is not None and (not value.is_finite() or not 0 <= value <= 1):
            raise ValueError(f"Invalid value: {identity}")
        data[identity] = value
    expected = set()
    for _, scope in SCOPES:
        for method in METHODS:
            run = key(method, scope)
            reg = registry[run]
            protocol = "coverage_cross_combination_v1" if scope == "cross" else "retrieval_v1"
            if (reg["protocol"] != protocol or reg["split"] != "test"
                    or reg["status"] != "report_hashes_verified"):
                raise ValueError(f"Unverified or wrong protocol: {run}")
            metrics = COVERAGE if scope == "cross" else EXACT + COVERAGE
            expected.update((run, g, metric, k) for g in GROUPS for metric in metrics for k in KS)
    if set(data) != expected:
        raise ValueError("Incomplete or unexpected metric coverage; update the report scope explicitly")

    def value(method, scope, group, metric, k):
        return data.get((key(method, scope), group, metric, k))

    def fmt(v):
        return "—" if v is None else f"{v * 100:.4f}"

    out = [f"数据校验：{len(registry)}项test评价，{len(data):,}条指标记录。源文件SHA-256：", ""]
    for name in ("metrics.csv", "registry.csv", "snapshot.json"):
        out.append(f"- `{name}`：`{hashlib.sha256((CATALOG / name).read_bytes()).hexdigest()}`")
    out.append("")
    out += table(["范围", "方法", "登记key／run_id", "报告来源"], [
        [label, NAMES[m], f"`{key(m, scope)}`<br>`{registry[key(m, scope)]['run_id']}`",
         registry[key(m, scope)]["report_sources"].replace(";", "<br>")]
        for label, scope in SCOPES for m in METHODS
    ])
    out += ["## 总体指标排序", "",
            "每项指标分别列出全部K的六方法分数及名次；括号内为同一行名次，越小越好。"
            "同值并列使用竞赛排名（例如1、1、3），排名按未四舍五入值计算。", ""]
    for label, scope in SCOPES:
        out += ["<details>", f"<summary>{label}：全部K逐指标排名</summary>", ""]
        metrics = COVERAGE if scope == "cross" else EXACT + COVERAGE
        for metric in metrics:
            out += [f"**{metric}**", ""]
            rows = []
            for k in KS:
                vs = [value(m, scope, "all", metric, k) for m in METHODS]
                cells = ["—" if v is None else
                         f"{fmt(v)}（{1 + sum(x is not None and x > v for x in vs)}）" for v in vs]
                rows.append([k, *cells])
            out += table(["K", *(NAMES[m] for m in METHODS)], rows)
        out += ["</details>", ""]
    out += ["## 全部输出指标", "",
            "以下按协议、split、group分别展开，逐方法给出全部K。"
            f"包含{len(data):,}条指标记录，不把缺失报告或无定义值填成0。FullCoverage的H/P/R/AP依次对应"
            "Hit、Precision、Recall、mAP；nDCG为Coverage-nDCG，MC为MeanCoverage。"
            "宏平均按原报告读取，不用总体值重新估算。", ""]
    for label, scope in SCOPES:
        metrics = COVERAGE if scope == "cross" else EXACT + COVERAGE
        headers = (["nDCG", "MC", "FC-H", "FC-P", "FC-R", "FC-AP"] if scope == "cross" else
                   ["exact-H", "exact-P", "exact-R", "exact-AP", "nDCG", "MC", "FC-H", "FC-P", "FC-R", "FC-AP"])
        for group, group_name in GROUPS.items():
            out += ["<details>", f"<summary>{label} · {group_name} · 六方法全部K</summary>", ""]
            rows = [[NAMES[m], k, *(fmt(value(m, scope, group, metric, k)) for metric in metrics)]
                    for m in METHODS for k in KS]
            out += table(["方法", "K", *headers], rows)
            out += ["</details>", ""]
    return "\n".join(out).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without modifying the document")
    args = parser.parse_args()
    current = DOCUMENT.read_text(encoding="utf-8")
    before, remainder = current.split(BEGIN)
    _, after = remainder.split(END)
    expected = before + BEGIN + "\n\n" + render() + END + after
    if args.check:
        if current != expected:
            raise SystemExit("Result tables are stale; run scripts/summarize_results.py")
        print("OK: test-only evaluations at K=1/5/10/20/50/100; generated tables match")
    else:
        DOCUMENT.write_text(expected, encoding="utf-8")
        print(f"Updated {DOCUMENT}")


if __name__ == "__main__":
    main()
