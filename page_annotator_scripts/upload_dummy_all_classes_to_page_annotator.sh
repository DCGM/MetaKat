# Usage: bash page_annotator_scripts/upload_dummy_all_classes_to_page_annotator.sh annotator_app/examples/add_image_from_folder.py <API_KEY>
API_ADD_IMAGE_FROM_FOLDER_SCRIPT=$1
API_KEY=$2
SRC_PATH=/mnt/matylda1/ikohut/data/smart_digiline/page_types/dummy_all_classes

python $API_ADD_IMAGE_FROM_FOLDER_SCRIPT --api_key $API_KEY --url https://page.semant.cz/api/ --list dummy_list --camera dummy_camera --path ${SRC_PATH}/images --annotations ${SRC_PATH}/dummy.json

