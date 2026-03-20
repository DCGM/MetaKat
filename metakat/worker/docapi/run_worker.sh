export WORKER_KEY=metakat.pNUDguoq.5ojmR7qVU207ttkMZF3mznsYHlLWDYDUgnt9ac62
export BASE_DIR=/home/ikohut/data/metakat_worker

export PYTHONPATH=/home/ikohut/Projects/MetaKat/libs/DocAPI:/home/ikohut/Projects/MetaKat/libs/detector-wrapper:/home/ikohut/Projects/MetaKat

source /home/ikohut/python_env/metakat/bin/activate 

python metakat_worker.py
