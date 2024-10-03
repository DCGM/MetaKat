PROJECT_DIR="/home/ikohut/Projects/digilinka"
METADATA_REPOSITORY="/mnt/matylda5/ibenes/projects/semant/metadata-repository"
DATASETS="cuni_fsv.public cuni_lf1.public cuni.public knav.private knav.public mzk.private"
MAPPINGS_DIR="/mnt/matylda1/ikohut/data/smart_digiline/page_types/mappings"

for i in $DATASETS; do
  echo $i
  python $PROJECT_DIR/page_type_datasets_from_mods/create_page_to_doc_mapping.py \
    --ids-jsonl $METADATA_REPOSITORY/$i.jsonl \
    --output-mapping-file $MAPPINGS_DIR/$i.page_to_doc_mapping;
done;