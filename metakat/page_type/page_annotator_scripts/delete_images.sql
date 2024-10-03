DELETE
FROM list_images
USING images
WHERE list_images.image_id = images.id
  and images.capture_time > '2024-06-03'::date;

DELETE
FROM list_objects
USING (
	SELECT * FROM images JOIN positions ON images.id = positions.image_id
	WHERE images.capture_time > '2024-06-03'::date
) pos
WHERE list_objects.object_id = pos.object_id;

DELETE
FROM positions
USING images
WHERE positions.image_id = images.id
  and images.capture_time > '2024-06-03'::date;

DELETE
FROM images
WHERE images.capture_time > '2024-06-03'::date;