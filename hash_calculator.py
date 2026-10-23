import hashlib


def calculate_hashes(video_path, chunk_size=1024 * 1024):
    """Return MD5 and SHA-256 of the file, read in a single pass (1 MB chunks)."""
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}