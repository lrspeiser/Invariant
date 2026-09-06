"""Prepare exact manifest-listed Git blobs without writing local Git metadata.

This helper performs no network requests or publication. The authorized GitHub
integration can upload these blobs, build on a verified remote tree, and perform
a non-forced ref update. Never create a root tree that discards unrelated files.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,subprocess
from pathlib import Path
from mond_atlas_common import ROOT,read_json,write_json,digest


def prepare(manifest_path,output,exclude):
    manifest=read_json(manifest_path)
    if output.exists():raise FileExistsError('immutable publication packet')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=manifest['base_commit']:raise ValueError('local base differs from manifest; review before preparing')
    listing=subprocess.check_output(['git','ls-tree','-rz','HEAD'],cwd=ROOT)
    tree={}
    for record in listing.split(b'\0'):
        if not record:continue
        metadata,path=record.split(b'\t',1);mode,kind,sha=metadata.decode().split()
        tree[path.decode('utf-8')]=(mode,sha)
    items=list(manifest['files'])+[dict(path=manifest_path.relative_to(ROOT).as_posix(),sha256=digest(manifest_path),bytes=manifest_path.stat().st_size)]
    output.mkdir(parents=True);entries=[];unchanged=[]
    for item in items:
        relative=item['path']
        if relative in exclude:continue
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or not relative.startswith(('scripts/','tests/','configs/','docs/','work/gravity-first-principles/')):
            raise ValueError('publication path outside intended scope')
        if path.suffix.lower() in ('.fits','.npz','.npy') or '/private/' in relative:raise ValueError('private/raw array in publication manifest')
        data=path.read_bytes()
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('manifest content changed: '+relative)
        sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        mode=tree.get(relative,('100644',None))[0]
        if tree.get(relative,(None,None))[1]==sha:
            unchanged.append(relative);continue
        encoded=base64.b64encode(data).decode('ascii');index=len(entries)
        encoded_path=output/(str(index)+'.base64');encoded_path.write_text(encoded,encoding='ascii',newline='\n')
        entries.append(dict(index=index,path=relative,mode=mode,bytes=len(data),sha256=item['sha256'],
            git_blob_sha=sha,base64_characters=len(encoded),encoded_file=encoded_path.name))
    result=dict(status='PREPARED_NOT_PUBLISHED',repository='lrspeiser/Invariant',branch='main',base_commit=head,
        manifest_path=manifest_path.relative_to(ROOT).as_posix(),manifest_sha256=digest(manifest_path),
        excluded_paths=exclude,unchanged_base_paths=unchanged,entries=entries,
        blob_count=len(entries),total_content_bytes=sum(e['bytes'] for e in entries),
        requirement='Verify remote parent/tree, create a tree based on that tree, verify every returned blob SHA, and move main with force=false only after current-head review.')
    write_json(output/'packet.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='entries'}))


def read_chunk(packet_path,index,offset,count):
    if offset<0 or count<1 or count>256000:raise ValueError('invalid transfer chunk')
    packet=read_json(packet_path);entry=packet['entries'][index]
    path=packet_path.parent/entry['encoded_file']
    with path.open(encoding='ascii') as stream:
        stream.seek(offset);content=stream.read(count)
    print(json.dumps(dict(index=index,offset=offset,content=content,total=entry['base64_characters'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exclude',action='append',default=[])
    p=sub.add_parser('chunk');p.add_argument('--packet',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--offset',type=int,required=True);p.add_argument('--count',type=int,default=128000)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.manifest.resolve(),args.output.resolve(),args.exclude)
    else:read_chunk(args.packet.resolve(),args.index,args.offset,args.count)
