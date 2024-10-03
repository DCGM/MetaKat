PROJECT_DIR="/home/ikohut/Projects/digilinka"
METADATA_REPOSITORY="/mnt/matylda5/ibenes/projects/semant/metadata-repository"
DATASETS="cuni_fsv.public cuni_lf1.public cuni.public knav.private knav.public mzk.private"
MAPPINGS_DIR="/mnt/matylda1/ikohut/data/smart_digiline/page_types/mappings"
ANN_DIR="/mnt/matylda1/ikohut/data/smart_digiline/page_types/ann"
for i in $DATASETS; do
  echo $i
  if [[ $1 == "redo_mapping" ]]; then
    python $PROJECT_DIR/page_type_datasets_from_mods/create_neighbour_page_mapping.py \
      --ids-jsonl $METADATA_REPOSITORY/$i.jsonl \
      --output-mapping-file $MAPPINGS_DIR/$i.neighbour_page_mapping;
  fi
  cat $MAPPINGS_DIR/$i.neighbour_page_mapping | grep -Ff $ANN_DIR/pages.all.ids > $MAPPINGS_DIR/$i.neighbour_page_mapping.current
  cat $MAPPINGS_DIR/$i.neighbour_page_mapping.current | awk '{ printf "%s\n%s\n%s\n",$1,$2,$4 }' | grep -v None | sort | uniq > $MAPPINGS_DIR/$i.neighbour_page_mapping.current.ids
done;

for i in $MAPPINGS_DIR/ndk.*.neighbour_page_mapping; do
  echo $i
  cat $i | grep -Ff $ANN_DIR/pages.all.ids > $i.current
  cat $i.current | awk '{ printf "%s\n%s\n%s\n",$1,$2,$4 }' | grep -v None | sort | uniq > $i.current.ids
done;

cat $MAPPINGS_DIR/*.neighbour_page_mapping.current > $MAPPINGS_DIR/neighbour_page_mapping.current.all
cat $MAPPINGS_DIR/*.neighbour_page_mapping.current.ids > $MAPPINGS_DIR/neighbour_page_mapping.current.ids.all