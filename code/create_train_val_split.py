import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd

SOURCE_FILE = "data/train.parquet"
TRAIN_OUTPUT_FILE = "data/train_split.parquet"
VAL_OUTPUT_FILE = "data/val_split.parquet"

VALIDATION_SIZE = 0.2
BATCH_SIZE = 5000
sampling_ratio = 0.001
def create_train_val_split():
    """
    Method splits train -> train_split + val_split parquet files
    It avoids entire loading data into memory
    """

    if os.path.exists(TRAIN_OUTPUT_FILE) and os.path.exists(VAL_OUTPUT_FILE):
        print("Split files already exist. Skipping creation")
        return

    print(f"Reading source file: {SOURCE_FILE}")
    pq_file = pq.ParquetFile(SOURCE_FILE)

    num_rows = pq_file.metadata.num_rows
    print(f"Total rows in source file: {num_rows}")
    
    sample_size = int(num_rows * sampling_ratio)
    print(f"Sampling {sample_size} rows out of {num_rows} rows")
    # Create and shuffle all indices
    print("Creating and shuffling indices for the split...")

    all_indices = np.random.permutation(num_rows)
    split_point = int((1-VALIDATION_SIZE)*num_rows)

    train_indices = all_indices[:split_point]
    val_indices = all_indices[split_point:]

    # use sets for fast lookup
    train_indices_set = set(train_indices)
    val_indices_set = set(val_indices)
    
    print(f"Training set size: {len(train_indices_set)} rows")
    print(f"Validation set size: {len(val_indices_set)} rows")

    # Setup parquet writer for output files
    schema = pq_file.schema_arrow

    train_writer = pq.ParquetWriter(TRAIN_OUTPUT_FILE, schema=schema)
    val_writer = pq.ParquetWriter(VAL_OUTPUT_FILE, schema=schema)


    # Iterate through source file in batches and write to new files
    n_batches = int(sample_size / BATCH_SIZE)

    print(f"\nProcessing file in {n_batches} batches and split files...")
    num_batches = pq_file.num_row_groups

    try:
        for i, batch in enumerate(pq_file.iter_batches(batch_size=BATCH_SIZE)):
            print(f"Processing batch {i+1}...")

            batch_df = batch.to_pandas()

            # filter the batches for train and val rows
            train_mask = batch_df.index.isin(train_indices_set)
            val_mask = batch_df.index.isin(val_indices_set)

            train_df = batch_df[train_mask]
            val_df = batch_df[val_mask]

            if i == (n_batches - 1):
                break
            # write non empty batches to files
            if not train_df.empty:
                train_writer.write_table(pa.Table.from_pandas(train_df, schema=schema))
            if not val_df.empty:
                val_writer.write_table(pa.Table.from_pandas(val_df, schema=schema))
    finally:
        train_writer.close()
        val_writer.close()
        print("\nFinished splitting files:")
        print(f"Training data: {TRAIN_OUTPUT_FILE}")
        print(f"Validation data: {VAL_OUTPUT_FILE}")

if __name__ == "__main__":
    create_train_val_split()