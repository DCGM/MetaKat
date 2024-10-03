/* Deletes images from list 82 and 102 which are at the same time in another list */

DELETE FROM list_images WHERE concat(list_images.image_id, list_images.list_id) IN
(SELECT concat(list_images.image_id, list_images.list_id) AS pfk FROM list_images JOIN
	(SELECT list_images.image_id, array_agg(list_images.list_id)
	FROM list_images JOIN lists ON list_images.list_id=lists.id
	GROUP BY list_images.image_id
	HAVING array[82,102] <@ (array_agg(list_images.list_id))
	AND 3=array_length((array_agg(list_images.list_id)), 1)) myagg ON myagg.image_id=list_images.image_id
WHERE list_images.list_id=82 OR list_images.list_id=102);