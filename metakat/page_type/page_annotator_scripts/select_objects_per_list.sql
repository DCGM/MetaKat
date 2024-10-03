SELECT list_name AS type, COUNT(lists.id) AS total, COUNT(object_annotations_max.object_id) AS annotated FROM
	lists LEFT JOIN list_objects ON lists.id = list_objects.list_id
	LEFT JOIN (
	            SELECT MAX(object_annotations.timestamp), object_annotations.object_id
                FROM object_annotations
	            GROUP BY object_annotations.object_id
              ) object_annotations_max ON list_objects.object_id = object_annotations_max.object_id
	LEFT JOIN positions ON list_objects.object_id = positions.object_id
WHERE positions.ignore = FALSE
GROUP BY list_name ORDER BY list_name ASC;