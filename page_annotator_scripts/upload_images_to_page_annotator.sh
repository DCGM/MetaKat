# Usage: ./upload_images_to_page_annotator.sh annotator_app/examples/add_images_from_folder.py data/annotations/ API_KEY
API_ADD_IMAGE_FROM_FOLDER_SCRIPT=$1
SRC_PATH=$2
API_KEY=$3

for i in ${SRC_PATH}/*.json; do
	echo ${i};
	TYPE=$(basename ${i/.json/});
	IMAGES=${SRC_PATH}/images/${TYPE}
	echo $IMAGES
	python $API_ADD_IMAGE_FROM_FOLDER_SCRIPT --api_key $API_KEY --url https://page.semant.cz/api/ --list $TYPE --camera pages --path $IMAGES --annotations $i
done;
