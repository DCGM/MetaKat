DELETE FROM list_images WHERE list_images.image_id IN
(SELECT images.id FROM images LEFT JOIN positions ON images.id=positions.image_id WHERE positions.image_id IS NULL);

DELETE FROM image_annotations WHERE image_annotations.image_id IN
(SELECT images.id FROM images LEFT JOIN positions ON images.id=positions.image_id WHERE positions.image_id IS NULL);

DELETE FROM images WHERE images.id IN
(SELECT images.id FROM images LEFT JOIN positions ON images.id=positions.image_id WHERE positions.image_id IS NULL);