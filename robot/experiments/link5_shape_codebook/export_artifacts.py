"""Hash a focused review export; retain requested mask/input/prediction provenance."""
import argparse
import os
from pathlib import Path

from .common import read,sha,write
from .report import run as report


def keep(path):
    parts=path.parts
    if any(part in ('replay', 'reference_replay') for part in parts):
        return False
    if 'geometry-work' in parts or 'vendor-source-snapshot' in parts or 'foundationpose-debug' in parts:
        return False
    if path.suffix.lower()=='.mp4' and parts[:3]!=('full_video','replay','videos'):
        return False
    if path.suffix.lower() in ('.zip','.gz','.tmp','.pyc') or '__pycache__' in parts:
        return False
    if len(parts)>1 and parts[0] in ('training','training_robot_structural') and parts[1]=='link5' and path.name.startswith('epoch_'):
        return False
    if path.name in ('export_manifest.json','export_files.txt','collection_files.txt','collection_cache.json'):
        return False
    if path.name.startswith('postprocess-') and path.suffix=='.log':
        return False  # reporting may still be appending while this manifest is made
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();output=Path(read(args.config)['output'])
    report(output)
    scope_path=output/'full_video/metadata/evaluation_scope.json'
    selected=set(read(scope_path)['selected_video_ids']) if scope_path.exists() else None
    def selected_keep(relative):
        if not keep(relative):return False
        parts=relative.parts
        if selected:
            if len(parts)>=3 and parts[:2]==('full_video','cases') and parts[2] not in selected:return False
            if len(parts)>=4 and parts[:3]==('full_video','replay','data') and parts[3] not in selected:return False
            if len(parts)>=4 and parts[:3]==('full_video','replay','videos') and relative.stem not in selected:return False
        if str(relative)=='full_video/replay/preview.png':return False
        return True
    records=[]
    for directory,folders,files in os.walk(output,followlinks=False):
        folders[:]=sorted(name for name in folders if selected_keep((Path(directory)/name).relative_to(output)))
        for name in sorted(files):
            file=Path(directory)/name;relative=file.relative_to(output)
            if file.is_file() and not file.is_symlink() and selected_keep(relative):
                records.append(dict(path=str(relative),bytes=file.stat().st_size,sha256=sha(file)))
    metadata=output/'metadata'
    write(metadata/'export_manifest.json',dict(source_root=str(output),files=records,total_bytes=sum(row['bytes'] for row in records)))
    (metadata/'export_files.txt').write_text('\n'.join(row['path'] for row in records)+'\nmetadata/export_manifest.json\nmetadata/export_files.txt\n')
    print('LINK5_REVIEW_EXPORT_HASHED',len(records),sum(row['bytes'] for row in records),flush=True)


if __name__=='__main__':main()
