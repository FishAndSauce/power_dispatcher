import boto3
import json
from io import StringIO, BytesIO
import pandas as pd
from dataclasses import dataclass
import os
from time import time


@dataclass
class CacheStatus:
    call_st: str
    filepath: str
    exists: bool = None
    expired: bool = None

    @property
    def execute_call(self):
        if not self.exists:
            return True
        if self.expired:
            return True
        else:
            return False


@dataclass
class CacheManager:
    directory: str
    expiration: int
    dummy: bool = False
    _log_fp: str = None

    def __post_init__(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self._log_fp = os.path.join(self.directory, 'log.json')
        if not os.path.exists(self._log_fp):
            self._overwrite_log({'next_call_ref_n': 0})
        self._call_record = {}
        self._file_references = []

    def _retrieve_log(self):
        with open(self._log_fp, 'r') as f:
            file = json.load(f)
        return file

    def _overwrite_log(self, obj: dict):
        with open(self._log_fp, 'w') as f:
            json.dump(obj, f, indent=2)

    def _name_file(self):
        log = self._retrieve_log()
        name = str(log['next_call_ref_n']).zfill(10)
        log['next_call_ref_n'] += 1
        self._overwrite_log(log)
        return name

    def _exists(self, call_str: str) -> CacheStatus:
        if call_str in self._call_record.keys():
            fp = self._call_record[call_str]
            exists = os.path.exists(fp)
        else:
            fp = os.path.join(self.directory, self._name_file())
            exists = False
        return CacheStatus(call_str, fp, exists, False)

    def _expired(self, fp):
        return time() - os.path.getmtime(fp) > self.expiration

    def new_call_record(self, cs: CacheStatus):
        if not self.dummy:
            log = self._retrieve_log()
            log[cs.call_st] = cs.filepath
            self._overwrite_log(log)

    def cache(self, call_str: str) -> CacheStatus:
        if self.dummy:
            return CacheStatus(
                call_str,
                'dummy_to_trigger_call',
                False,
                True
            )
        status = self._exists(call_str)
        if status.exists:
            status.expired = self._expired(status.filepath)
        else:
            status.expired = True
        return status


@dataclass
class s3BucketManager:
    bucket: str
    aws_id: str
    aws_key: str
    cache_manager: CacheManager = None

    def __post_init__(self):
        if not self.cache:
            self.cache = CacheManager('dummy_cache', 1, dummy=True)

    @property
    def resource(self):
        return boto3.resource(
            's3',
            aws_access_key_id=self.aws_id,
            aws_secret_access_key=self.aws_key
        )

    @staticmethod
    def path(folders, fn):
        folders_str = '/'.join(folders)
        return f'{folders_str}/{fn}'

    def df_to_s3_csv(
            self, 
            df: pd.DataFrame, 
            folders, 
            fn, 
            **kwargs
    ):
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, **kwargs)
        self.resource.Object(
            self.bucket,
            self.path(folders, fn),
        ).put(Body=csv_buffer.getvalue())

    def dict_to_s3_json(
            self,
            json_data: dict,
            folders,
            fn,
            indent=2
    ):
        self.resource.Object(
            self.bucket,
            self.path(folders, fn),
        ).put(Body=bytes(
            json.dumps(
                json_data,
                indent=indent
            ).encode('UTF-8')))

    def s3_json_to_dict(
            self,
            folders,
            fn
    ) -> dict:
        file_object = self.resource.Object(
            self.bucket,
            self.path(folders, fn),
        )
        file = file_object.get()['Body'].read().decode('utf-8')
        json_data = json.loads(file)
        return json_data

    def df_to_s3_ftr(
            self,
            df: pd.DataFrame,
            folders,
            fn,
            **kwargs
    ):
        buffer = BytesIO()
        df.to_feather(buffer, **kwargs)
        self.resource.Object(
            self.bucket,
            self.path(folders, fn),
        ).put(Body=buffer.getvalue())

    def s3_ftr_to_df(
            self,
            folders,
            fn
    ) -> pd.DataFrame:
        file_object = self.resource.Object(
            self.bucket,
            self.path(folders, fn)
        )
        file = BytesIO(file_object.get()['Body'].read())
        return pd.read_feather(file)

    def s3_csv_to_df(
            self,
            folders,
            fn,
    ) -> pd.DataFrame:
        path = self.path(folders, fn)
        cache_status = self.cache_manager.cache(path)
        if cache_status.execute_call:
            file_object = self.resource.Object(
                self.bucket,
                path
            )
            file = BytesIO(file_object.get()['Body'].read())
            df = pd.read_csv(file)
            df.to_feather(cache_status.filepath)
            self.cache_manager.new_call_record(cache_status)
            return df
        else:
            return pd.read_feather(cache_status.filepath)