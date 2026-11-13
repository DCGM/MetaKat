# Development worker

Checkout: `~/Projects/MetaKat.worker.dev`, branch `issue-13-docapi-v1.2.1`. The live worker
uses `~/Projects/MetaKat.worker` on branch `worker`. These are separate working
directories sharing Git history. Edits to development source files do not alter
the live checkout.

`run_worker_dev.sh` connects only to `http://127.0.0.1:9999` (or `DOCAPI_PORT`)
using the local DocAPI checkout in `libs/DocAPI`. It uses
`~/python_env/metakat-dev`, a development virtual environment which
reads the existing libraries from `~/python_env/metakat` and keeps extra
dependencies (including `email-validator`) separate. The live environment is unchanged.
`PYTHONPATH` selects this development checkout, its own text-geometry-aligner
checkout, and `libs/DocAPI` for the API adapter/wrapper. Do not reinstall
editable packages into the live environment to switch checkouts.
It does not modify or restart the live worker.

The local DocAPI worker key is in the ignored `.docapi_dev_worker_key` beside the
script, with permissions `600`. It belongs to the `metakat-worker-dev` worker
account on the local API. After resetting the local database, create a new key
with role `worker` through the local API's admin endpoints and replace this file.
Never use the live worker key here.

Jobs are under `/mnt/kolosus/data/metakat_worker_dev/jobs/`. Per-job logs and
inspection outputs stay with their job directories on kolosus. Downloaded
engines are on NVMe in `~/data/metakat_worker_dev/engines/`, and general worker
logs are in `~/data/metakat_worker_dev/logs/`; the worker lock is also on NVMe.
This matches the live worker's storage layout with separate development paths.
The launcher retains job files and PDF/MODS inspection outputs. Email uses `smtp.fit.vut.cz:25` with STARTTLS and sends to
`ikohut@fit.vut.cz`. Startup sends an alive report immediately; periodic reports
run every 24 hours. `DEV_MAIL_*` variables can override the relay, sender, recipients,
timeout and alive interval. Optional credentials use `DEV_MAIL_PASSWORD_FILE`
and `DEV_MAIL_USERNAME`; they are independent of the live worker.
Mail subjects name the worker `MetaKat worker dev` (override with
`DEV_WORKER_IDENTITY`), e.g. `[MetaKat worker dev] ALIVE (pckohut)`, so they are
not confused with the production worker's `MetaKat worker prod`.
Attempt history is appended to `~/data/metakat_worker_dev/logs/job_history.jsonl`.
The sibling `job_history.stats.json` caches cumulative statistics and recent
records for rolling windows. It can be rebuilt from the JSONL history.
`DEV_JOB_HISTORY_PATH` overrides this path in the dev launcher; an empty value
disables history. Other deployments use `JOB_HISTORY_PATH`. Startup
and periodic health emails use the same alive report, including all-time,
7-day and 24-hour attempt totals and unique-job counts.
The default VLM key is `.openrouter_api_key_naki-tools`, loaded into
`OPENROUTER_API_KEY`. Explicit keys supplied by engines or jobs take precedence
as usual. The launcher requires a readable, nonempty default key file.
Starting an idle worker
does not make inference requests. Local VLM engines share this machine's GPU
with the live worker, so avoid running competing GPU workloads.

Start a detached screen session from this checkout:

```sh
cd ~/Projects/MetaKat.worker.dev/metakat/worker/docapi
screen -dmS metakat_worker_dev ./run_worker_dev.sh
screen -r metakat_worker_dev
```

Detach with Ctrl-A, D. Stop with Ctrl-C while attached. To load code changes,
stop the development worker and start the script again. The launcher prevents
duplicate development workers with a file lock.

Inspect logs with:

```sh
tail -f ~/data/metakat_worker_dev/logs/worker.log
```
