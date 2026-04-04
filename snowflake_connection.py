from pathlib import Path

import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization


def get_snowflake_connection(
    user: str = "BLUEJAY",
    account: str = "pgb87192",
    warehouse: str = "COMPUTE_WH",
    database: str = "JOBPREP_DB",
    schema: str = "RAW_DATA",
    private_key_path: str = "rsa_key.p8",
):
    key_path = Path(private_key_path)
    if not key_path.is_absolute():
        key_path = Path.cwd() / key_path

    with open(key_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend(),
        )

    pkb = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return snowflake.connector.connect(
        user=user,
        account=account,
        private_key=pkb,
        warehouse=warehouse,
        database=database,
        schema=schema,
    )
