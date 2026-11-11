"""Neon S3-compatible Object Storage client and helpers."""

from __future__ import annotations

import logging

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from hybrid_rag.settings import settings

logger = logging.getLogger(__name__)

_s3_client = None


def is_storage_configured() -> bool:
    """Return True if S3 credentials and endpoint are present."""
    return bool(
        settings.aws_endpoint_url_s3
        and settings.aws_access_key_id
        and settings.aws_secret_access_key
    )


def get_s3_client():
    """Return a singleton boto3 S3 client configured for Neon Object Storage."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client

    if not is_storage_configured():
        logger.debug("Neon Object Storage credentials/endpoint not configured")
        return None

    try:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=settings.aws_endpoint_url_s3,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
            config=Config(
                s3={"addressing_style": "path"},
                signature_version="s3v4",
                connect_timeout=5,
                read_timeout=10,
                retries={"max_attempts": 2},
            ),
        )
        return _s3_client
    except Exception as exc:
        logger.warning("Failed to initialize S3 client: %s", exc)
        return None


def reset_storage_client() -> None:
    """Reset the cached storage client (used in tests)."""
    global _s3_client
    _s3_client = None


def check_storage_connection() -> bool:
    """Validate connectivity to the configured Neon Object Storage bucket."""
    client = get_s3_client()
    if client is None:
        return False
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
        return True
    except ClientError as exc:
        # Fall back to listing 1 object in case head_bucket permission differs
        try:
            client.list_objects_v2(Bucket=settings.s3_bucket, MaxKeys=1)
            return True
        except Exception:
            logger.warning("Storage connection check failed: %s", exc)
            return False
    except Exception as exc:
        logger.warning("Storage connection check failed: %s", exc)
        return False


def upload_file(
    key: str,
    data: bytes,
    content_type: str = "application/octet-stream",
    bucket: str | None = None,
) -> str | None:
    """Upload binary data to object storage under the specified key."""
    client = get_s3_client()
    if client is None:
        return None
    target_bucket = bucket or settings.s3_bucket
    try:
        client.put_object(
            Bucket=target_bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        logger.info("Uploaded %d bytes to s3://%s/%s", len(data), target_bucket, key)
        return key
    except Exception as exc:
        logger.error("Failed to upload object to s3://%s/%s: %s", target_bucket, key, exc)
        raise


def download_file(key: str, bucket: str | None = None) -> bytes | None:
    """Download binary data from object storage."""
    client = get_s3_client()
    if client is None:
        return None
    target_bucket = bucket or settings.s3_bucket
    try:
        response = client.get_object(Bucket=target_bucket, Key=key)
        return response["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        logger.error("Failed to download object from s3://%s/%s: %s", target_bucket, key, exc)
        raise
    except Exception as exc:
        logger.error("Failed to download object from s3://%s/%s: %s", target_bucket, key, exc)
        raise


def delete_file(key: str, bucket: str | None = None) -> bool:
    """Delete an object from object storage."""
    client = get_s3_client()
    if client is None:
        return False
    target_bucket = bucket or settings.s3_bucket
    try:
        client.delete_object(Bucket=target_bucket, Key=key)
        logger.info("Deleted object s3://%s/%s", target_bucket, key)
        return True
    except Exception as exc:
        logger.warning("Failed to delete object s3://%s/%s: %s", target_bucket, key, exc)
        return False


def generate_presigned_url(
    key: str,
    expires_in: int = 3600,
    bucket: str | None = None,
) -> str | None:
    """Generate a presigned GET URL for an object."""
    client = get_s3_client()
    if client is None:
        return None
    target_bucket = bucket or settings.s3_bucket
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": target_bucket, "Key": key},
            ExpiresIn=expires_in,
        )
    except Exception as exc:
        logger.warning(
            "Failed to generate presigned URL for s3://%s/%s: %s", target_bucket, key, exc
        )
        return None
