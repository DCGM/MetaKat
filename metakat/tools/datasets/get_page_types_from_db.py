import argparse
import logging
import os
import shutil
import sys
import time

from sqlalchemy import create_engine, MetaData, select

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--db', type=str, required=True)
    parser.add_argument('--library', type=str, required=True)
    parser.add_argument('--page-type', type=str, required=True)
    parser.add_argument('--public', action='store_true')
    parser.add_argument('--output-dir', type=str, required=True)

    parser.add_argument('--logging-level', default=logging.INFO)
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    log_formatter = logging.Formatter('PARSE PATH - %(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    db_engine = create_engine(args.db)

    db_model = MetaData()
    db_model.reflect(bind=db_engine)
    db_model = db_model.tables

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'mods'), exist_ok=True)

    with db_engine.connect() as db_connection:
        query = select(db_model['meta_records'])
        query = query.where(db_model['meta_records'].c.library == args.library)
        query = query.where(db_model['meta_records'].c.public == args.public)
        query = query.where(db_model['meta_records'].c.image_path != 'not_found')
        query = query.where(db_model['meta_records'].c.image_path.isnot(None))
        query = query.where(db_model['meta_records'].c.page_type == args.page_type)
        query = query.where(db_model['meta_records'].c.depth == 1)
        db_meta_records = list(db_connection.execute(query).all())

        logger.info(f'Copying images and mods for {len(db_meta_records)} records')

        for i, db_meta_record in enumerate(db_meta_records):
            image_src_path = str(db_meta_record.image_path)
            image_src_path = str(os.path.join(*image_src_path.split('/')[:-1], f'uuid:{image_src_path.split("/")[-1]}'))
            image_src_path = '/' + image_src_path
            if os.path.exists(image_src_path):
                image_dst_path = str(os.path.join(args.output_dir,
                                     'images',
                                     f'{db_meta_record.parent_id}.{db_meta_record.order}.{db_meta_record.id}{os.path.splitext(db_meta_record.image_path)[1]}'))
                shutil.copy2(image_src_path, image_dst_path)
            else:
                logger.warning(f'Image not found: {image_src_path}')

            parent_mods_path_query = select(db_model['meta_records'].c.mods_path)
            parent_mods_path_query = parent_mods_path_query.where(db_model['meta_records'].c.id == db_meta_record.parent_id)
            parent_mods_path = db_connection.execute(parent_mods_path_query).scalar()
            mods_src_path = str(parent_mods_path)
            if os.path.exists(mods_src_path):
                mods_dst_path = str(os.path.join(args.output_dir,
                                     'mods',
                                     f'{db_meta_record.parent_id}.{db_meta_record.order}.{db_meta_record.id}{os.path.splitext(db_meta_record.mods_path)[1]}'))
                shutil.copy2(mods_src_path, mods_dst_path)
            else:
                logger.warning(f'Mods not found: {mods_src_path}')

            if (i + 1) % 100 == 0:
                logger.info(f'{i + 1}/{len(db_meta_records)} records processed')

        logger.info(f'{len(db_meta_records)}/{len(db_meta_records)} records processed')

if __name__ == '__main__':
    main()

