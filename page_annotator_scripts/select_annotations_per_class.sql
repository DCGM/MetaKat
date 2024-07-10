SELECT OA.value AS type, COUNT(OA.value) AS annotated FROM
    (
        SELECT MAX(object_annotations.id) AS oam_max_id , object_id
        FROM object_annotations
        GROUP BY object_id
    ) OAM
    JOIN object_annotations AS OA ON OAM.oam_max_id = OA.id
    LEFT JOIN positions ON OA.object_id = positions.object_id
    JOIN images ON positions.image_id = images.id
    JOIN cameras ON images.camera_id = cameras.id
WHERE positions.ignore = FALSE AND OA.value != '' AND cameras.uuid = 'pages'
GROUP BY OA.value ORDER BY OA.value ASC;