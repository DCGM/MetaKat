/* Deletes objects from list 82 and 102 which are at the same time in another list */

DELETE FROM list_objects WHERE concat(list_objects.object_id, list_objects.list_id) IN
(SELECT concat(list_objects.object_id, list_objects.list_id) AS pfk FROM list_objects JOIN
	(SELECT list_objects.object_id, array_agg(list_objects.list_id)
	FROM list_objects JOIN lists ON list_objects.list_id=lists.id
	GROUP BY list_objects.object_id
	HAVING array[82,102] <@ (array_agg(list_objects.list_id))
	AND 3=array_length((array_agg(list_objects.list_id)), 1)) myagg ON myagg.object_id=list_objects.object_id
WHERE list_objects.list_id=82 OR list_objects.list_id=102);

DELETE FROM list_objects WHERE concat(list_objects.object_id, list_objects.list_id) IN
(SELECT concat(list_objects.object_id, list_objects.list_id) AS pfk FROM list_objects JOIN
	(SELECT list_objects.object_id, array_agg(list_objects.list_id)
	FROM list_objects JOIN lists ON list_objects.list_id=lists.id
	GROUP BY list_objects.object_id
	HAVING array[82] <@ (array_agg(list_objects.list_id))
	AND 2=array_length((array_agg(list_objects.list_id)), 1)) myagg ON myagg.object_id=list_objects.object_id
WHERE list_objects.list_id=82);

DELETE FROM list_objects WHERE list_objects.list_id=102