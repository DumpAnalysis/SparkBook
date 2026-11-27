"""Verify complete artifacts and resume pinned byte ranges in the WSL lab."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import urllib.parse
import urllib.request

CHUNK = 64 * 1024**2


@contextmanager
def locked_directory(directory):
    import fcntl
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def verify(path, size, sha256=None, git_blob=None):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ValueError(f'Wrong file type or size: {path}')
    digest = hashlib.sha256()
    blob = hashlib.sha1(f'blob {size}\0'.encode()) if git_blob else None
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
            if blob is not None:
                blob.update(block)
    if not (sha256 or git_blob) or (sha256 and digest.hexdigest() != sha256) or (blob and blob.hexdigest() != git_blob):
        raise ValueError(f'Publisher hash mismatch: {path}. Retain the file for inspection and move it aside before retrying.')
    return digest.hexdigest()


def fetch_pinned(url, path, size, sha256=None, git_blob=None):
    """Call with the destination directory locked. Publish only after identity verification."""
    if path.exists() or path.is_symlink():
        return verify(path, size, sha256, git_blob)
    partial = path.with_name(path.name + '.partial')
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise ValueError(f'Invalid partial file: {partial}')
    with partial.open('ab') as output:
        if output.tell() > size:
            raise ValueError(f'Partial file exceeds pinned size: {partial}')
        while output.tell() < size:
            start = output.tell()
            end = min(size, start + CHUNK) - 1
            # Some registries cache signed redirects by URL, including the byte range.
            separator = '&' if urllib.parse.urlsplit(url).query else '?'
            ranged_url = f'{url}{separator}spark_range={start}-{end}'
            request = urllib.request.Request(ranged_url, headers={'Range':f'bytes={start}-{end}', 'Accept-Encoding':'identity'})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.headers.get('Content-Encoding','identity').lower() != 'identity':
                    raise ValueError('Expected unencoded file bytes')
                if response.status == 206 and response.headers.get('Content-Range') == f'bytes {start}-{end}/{size}':
                    count = end-start+1
                elif response.status == 200 and start == 0 and not response.headers.get('Content-Range'):
                    count = size
                else:
                    raise ValueError(f'Unexpected response to byte range {start}-{end}. The existing partial file is retained.')
                length = response.headers.get('Content-Length')
                if length is not None and int(length) != count:
                    raise ValueError('Response length differs from the pinned range')
                while count:
                    block = response.read(min(1024**2, count))
                    if not block:
                        raise OSError(f'Interrupted transfer at byte {output.tell()}. Rerun the same download to resume.')
                    output.write(block)
                    count -= len(block)
                if response.read(1):
                    raise ValueError('Unexpected bytes after the pinned range')
    digest = verify(partial, size, sha256, git_blob)
    partial.replace(path)
    return digest
