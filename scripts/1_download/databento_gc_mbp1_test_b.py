# 下载测试集 B 的 MBP-1（GC 主力合约，2026-09-29 至 2026-11-19，估价 $12.98，从账户扣费）
# 与 databento_gc_mbp1.py 相同的批量任务设置，只是时间范围不同（config.yaml 的 data.test_b_start / test_b_end）
# 任务 ID 记在 GC_mbp1/test_b_job_id.txt：重新运行时沿用已提交的任务，不会重复扣费
# 测试集 B 只在 M6 最终评估时使用一次：下载后不打开、不统计、不画图
import time
import databento as db
from gold_dl.config import CFG, path

key = open(path("databento_key")).read().strip()
client = db.Historical(key)
OUT_DIR = str(path("raw_mbp1"))
ID_FILE = path("raw_mbp1", "test_b_job_id.txt")

if ID_FILE.is_file():
    job_id = ID_FILE.read_text().strip()
else:
    job = client.batch.submit_job(
        dataset=CFG["data"]["dataset"], schema="mbp-1",
        symbols=CFG["data"]["symbol"], stype_in="continuous",
        start=CFG["data"]["test_b_start"], end=CFG["data"]["test_b_end"],
        encoding="dbn", compression="zstd", split_duration="month",
    )
    job_id = job["id"]
    ID_FILE.write_text(job_id)
    print("submitted job", job_id, "cost", job.get("cost_usd"), flush=True)

while True:
    jobs = {j["id"]: j for j in client.batch.list_jobs(states="queued,processing,done,expired")}
    state = jobs.get(job_id, {}).get("state")
    print(time.strftime("%H:%M:%S"), "state:", state, "cost:", jobs.get(job_id, {}).get("cost_usd"), flush=True)
    if state == "done":
        break
    if state == "expired":
        raise SystemExit("job expired")
    time.sleep(60)

files = client.batch.download(job_id=job_id, output_dir=OUT_DIR)
print("downloaded", len(files), "files to", OUT_DIR, flush=True)
