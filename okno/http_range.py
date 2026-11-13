"""Seekable public HTTP reader for rasterio's opener, using requests CA/proxy trust."""
import io
from collections import OrderedDict
import requests


class RangeReader(io.RawIOBase):
    def __init__(self,url,mode='rb'):
        super().__init__()
        if mode not in ('r','rb'):raise ValueError('Только чтение')
        if not str(url).startswith('https://'):raise FileNotFoundError(url)
        self.url=url;self.pos=0;self.block_size=256*1024;self.blocks=OrderedDict()
        self.session=requests.Session()
        with self.session.get(url,headers={'Range':'bytes=0-0'},timeout=45) as r:
            r.raise_for_status()
            if r.status_code!=206:raise ValueError('Источник не поддерживает частичное чтение')
            self.size=int(r.headers['Content-Range'].split('/')[-1])
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        dest=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset if whence==2 else -1
        if dest<0:raise ValueError('Некорректное смещение')
        self.pos=dest;return dest
    def read(self,size=-1):
        end=self.size if size<0 else min(self.size,self.pos+size)
        chunks=[]
        while self.pos<end:
            block=self.pos//self.block_size
            if block not in self.blocks:
                start=block*self.block_size;stop=min(self.size,start+self.block_size)-1
                with self.session.get(self.url,headers={'Range':f'bytes={start}-{stop}'},timeout=45) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range','').split('/')[0]!=f'bytes {start}-{stop}':
                        raise ValueError('Источник вернул неверный диапазон байтов')
                    value=r.content
                    if len(value)!=stop-start+1:raise ValueError('Неполный диапазон байтов')
                self.blocks[block]=value
                if len(self.blocks)>32:self.blocks.popitem(last=False)
            data=self.blocks[block];offset=self.pos%self.block_size;n=min(end-self.pos,len(data)-offset)
            if n<=0:raise ValueError('Неполное чтение растра')
            chunks.append(data[offset:offset+n]);self.pos+=n
        return b''.join(chunks)
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)
    def close(self):
        if hasattr(self,'session'):self.session.close()
        super().close()
