PROJECT_DIR="/home/ikohut/Projects/digilinka"
METADATA_REPOSITORY="/mnt/matylda5/ibenes/projects/semant/metadata-repository"
DATASETS="cuni_fsv.public cuni_lf1.public cuni.public knav.private knav.public mzk.private"
PRIVATE_IMAGES_DIR="/mnt/library-downloads/"
PUBLIC_IMAGES_DIR="/mnt/matylda0/ihradis/knihovny_images_new_downloads/"
PUBLIC_IMAGES_DIR_2="/mnt/matylda0/ihradis/knihovny_images_new_downloads_2"
MAPPINGS_DIR="/mnt/matylda1/ikohut/data/smart_digiline/page_types/mappings"
OUTPUT_DIR="/mnt/matylda1/ikohut/data/smart_digiline/page_types/rev_tmp"
ANN_DIR="/mnt/matylda1/ikohut/data/smart_digiline/page_types/ann"

if [[ $1 == "redo_mapping" ]]; then
  for i in $DATASETS; do
    echo $i
    cat $MAPPINGS_DIR/$i.neighbour_page_mapping.current.ids | rev | cut -c 5- | rev > $MAPPINGS_DIR/$i.neighbour_page_mapping.current.ids.tmp
    cat $MAPPINGS_DIR/$i.page_to_doc_mapping | grep -Ff $MAPPINGS_DIR/$i.neighbour_page_mapping.current.ids.tmp > $MAPPINGS_DIR/$i.page_to_doc_mapping.current
    rm $MAPPINGS_DIR/$i.neighbour_page_mapping.current.ids.tmp
  done;
fi

python $PROJECT_DIR/page_type_datasets_from_mods/create_symlinks_to_images.py \
  --pages $MAPPINGS_DIR/mzk.private.neighbour_page_mapping.current.ids \
  --page-to-doc-mapping $MAPPINGS_DIR/mzk.private.page_to_doc_mapping.current \
  --images-dir $PRIVATE_IMAGES_DIR/mzk \
  --output-dir $OUTPUT_DIR

python $PROJECT_DIR/page_type_datasets_from_mods/create_symlinks_to_images.py \
  --pages $MAPPINGS_DIR/knav.private.neighbour_page_mapping.current.ids \
  --page-to-doc-mapping $MAPPINGS_DIR/knav.private.page_to_doc_mapping.current \
  --images-dir $PRIVATE_IMAGES_DIR/knav \
  --output-dir $OUTPUT_DIR

python $PROJECT_DIR/page_type_datasets_from_mods/create_symlinks_to_images.py \
  --pages $MAPPINGS_DIR/knav.public.neighbour_page_mapping.current.ids \
  --page-to-doc-mapping $MAPPINGS_DIR/knav.public.page_to_doc_mapping.current \
  --images-dir $PUBLIC_IMAGES_DIR_2/knav \
  --output-dir $OUTPUT_DIR

python $PROJECT_DIR/page_type_datasets_from_mods/create_symlinks_to_images.py \
  --pages $MAPPINGS_DIR/cuni.public.neighbour_page_mapping.current.ids \
  --page-to-doc-mapping $MAPPINGS_DIR/cuni.public.page_to_doc_mapping.current \
  --images-dir $PUBLIC_IMAGES_DIR_2/cuni \
  --output-dir $OUTPUT_DIR

python $PROJECT_DIR/page_type_datasets_from_mods/create_symlinks_to_images.py \
  --pages $MAPPINGS_DIR/cuni_fsv.public.neighbour_page_mapping.current.ids \
  --page-to-doc-mapping $MAPPINGS_DIR/cuni_fsv.public.page_to_doc_mapping.current \
  --images-dir $PUBLIC_IMAGES_DIR_2/cuni_fsv \
  --output-dir $OUTPUT_DIR

python $PROJECT_DIR/page_type_datasets_from_mods/create_symlinks_to_images.py \
  --pages $MAPPINGS_DIR/cuni_lf1.public.neighbour_page_mapping.current.ids \
  --page-to-doc-mapping $MAPPINGS_DIR/cuni_lf1.public.page_to_doc_mapping.current \
  --images-dir $PUBLIC_IMAGES_DIR/cuni_lf1 \
  --output-dir $OUTPUT_DIR