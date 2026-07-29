export WORKER_KEY=
export BASE_DIR=/mnt/kolosus/data/metakat_worker
export ENGINES_DIR=/home/ikohut/data/metakat_worker/engines
export LOGGING_DIR=/home/ikohut/data/metakat_worker/logs

export PYTHONPATH=/home/ikohut/Projects/MetaKat/libs/DocAPI:/home/ikohut/Projects/MetaKat/libs/detector-wrapper:/home/ikohut/Projects/MetaKat

source /home/ikohut/python_env/metakat/bin/activate 

python metakat_worker.py
