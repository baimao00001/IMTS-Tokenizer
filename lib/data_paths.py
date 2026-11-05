"""Read existing datasets in place; keep newly generated data in the run tree."""

from pathlib import Path
import tarfile

from lib.run_paths import resolve_output_dir


ROOT = Path(__file__).resolve().parents[1]


def dataset_cache_root(dataset, output_root=None):
    if output_root is None:
        return resolve_output_dir(ROOT) / "data_cache" / dataset
    return resolve_output_dir(ROOT, output_root)


def processed_path(input_root, cache_root, filename):
    existing = Path(input_root) / "processed" / filename
    return existing if existing.is_file() else cache_path(cache_root, Path("processed") / filename)


def cache_path(cache_root, child):
    return resolve_output_dir(ROOT, Path(cache_root) / child)


def extract_dataset_archive(archive, destination):
    destination = Path(destination).resolve()
    members = archive.getmembers()
    for member in members:
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination) or member.issym() or member.islnk():
            raise ValueError("Dataset archive contains an unsafe output path: " + member.name)
        if not (member.isfile() or member.isdir()):
            raise ValueError("Dataset archive contains an unsupported file type: " + member.name)
    # Members are validated above; avoid version-specific extraction defaults.
    options = {"filter": lambda member, _: member} if hasattr(tarfile, "data_filter") else {}
    archive.extractall(destination, members=members, **options)
