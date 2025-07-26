import os

import xgboost
import pyarrow.parquet as pq
import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

TRAIN_FILE_PATH = "data/train_split.parquet"
VAL_FILE_PATH  = "data/val_split.parquet"

TARGET_COLUMN = "selected"

BATCH_SIZE = 5_000

## RMM setup
print("Setting up RAPIDS Memory Manager (RMM)...")
mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource())
rmm.mr.set_current_device_resource(mr)
cp.cuda.set_allocator(rmm_cupy_allocator)
print("RMM setup complete.")

## this class defines how to load parquet data in chunks

class ParquetIterator(xgboost.DataIter):
    def __init__(self, file_path: str, target_name: str, batch_size: int):
        self._file_path = file_path
        self._target_name = target_name
        self._batch_size = batch_size

        # open the parquet file using pyarrow, this doesn't load into memory
        self._parquet_file = pq.ParquetFile(self._file_path)

        # xgb will create cache files, the code will create a directory if not present
        cache_dir = "./xgb_cache"
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        
        super().__init__(cache_prefix=os.path.join(cache_dir, "cache"))
        
    def reset(self) -> None:
        """
        This method is called by xgb to restart the ITERATION PROCESS.
        """

        self._batch_iterator = self._parquet_file.iter_batches(batch_size=self._batch_size)
    
    def next(self, input_data: callable) -> bool:
        """
        CORE METHOD:
        xgb calls it repeatedly to get batches
        """

        try:
            batch_df = next(self._batch_iterator).to_pandas()

            y = batch_df[self._target_name]
            X = batch_df.drop(columns=[self._target_name,"requestDate"])

            # move the data to GPU
            X_gpu = cp.asarray(X)
            y_gpu = cp.asarray(y)

            # # pass the GPU data to xgb
            # input_data(data=X_gpu, label=y_gpu) # type: ignore
            # # return True to indicate that there is more data
            return True

        except StopIteration:
            # return False to stop when data is exhausted
            return False
        

print("\nStarting XBGBoost training process...")
with xgboost.config_context(use_rmm=True, verbosity=3):
    # initialize custom ParquetIterator
    it_train = ParquetIterator(file_path=TRAIN_FILE_PATH, target_name=TARGET_COLUMN, batch_size=BATCH_SIZE)
    it_val = ParquetIterator(file_path=VAL_FILE_PATH, target_name=TARGET_COLUMN, batch_size=BATCH_SIZE)

    # Create a speacial training DMatrix for external memory on GPU
    print("Initializing ExtMemQuantileDMatrix. This may take some time on the first run...")
    Xy_train_dmatrix = xgboost.ExtMemQuantileDMatrix(it_train)
    print("Training DMatrix initialization complete.")

    # Create a speacial validation DMatrix
    print("Initializing ExtMemQuantileDMatrix. This may take some time on the first run...")
    Xy_val_dmatrix = xgboost.ExtMemQuantileDMatrix(it_val)
    print("Validation DMatrix initialization complete.")
    
    # Define the XGBoost parameters
    params = {
        'device': 'cuda',               # CRITICAL: Use the GPU.
        'tree_method': 'hist',          # Required for this DMatrix type.
        'objective': 'binary:logistic', # Example for classification.
        'eval_metric': 'auc',
        'grow_policy': 'depthwise',     # Recommended for external memory.
    }

    print("\nTraining the model...")

    bst = xgboost.train(
        params=params,
        dtrain=Xy_train_dmatrix,
        num_boost_rounds = 1000,
        evals=[(Xy_train_dmatrix,"train"), (Xy_val_dmatrix,"validation")],
        early_stopping_rounds=50,
        verbose_eval=25
    )


    print("\nTraining finished successfully!")
    print(f"Best iteration: {bst.best_iteration}, Best validation AUC: {bst.best_score}")

