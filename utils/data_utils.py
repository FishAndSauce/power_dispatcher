import boto3
from boto3 import s3
import json
from io import StringIO, BytesIO
import pandas as pd
from dataclasses import dataclass
from os.path import join
from typing import Type

from local_data.local_data import aws_credentials


@dataclass
class s3BucketManager:
    bucket: str
    aws_id: str = aws_credentials['id']
    aws_key: str = aws_credentials['access_key']
    
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
            fn
    ) -> pd.DataFrame:
        file_object = self.resource.Object(
            self.bucket,
            self.path(folders, fn)
        )
        file = BytesIO(file_object.get()['Body'].read())
        return pd.read_csv(file)
