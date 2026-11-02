# 下载 GC 主力合约一年 MBP-1（估价 $71.60，从免费额度扣）
# 用批量任务（batch job）：服务器打包好后按月分文件下载，适合大数据量
import time
import databento as db
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

key = open(ROOT / "databento_key.txt").read().strip()
OUT_DIR = str(ROOT / "GC_mbp1")

# 已提交过的任务 ID；为空时才提交新任务（避免重复扣费）
JOB_ID = "GLBX-20261025-PT9C6P3YKJ"

if JOB_ID:
    job_id = JOB_ID
else:
  job = client.batch.submit_job(
    dataset="GLBX.MDP3", schema="mbp-1",
    symbols="GC.v.0", stype_in="continuous",
    start="2025-09-28", end="2026-09-29",
    encoding="dbn", compression="zstd", split_duration="month",
  )
  job_id = job["id"]
  print("submitted job", job_id, flush=True)

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
for f in sorted(files):
    print(" ", f, flush=True)
