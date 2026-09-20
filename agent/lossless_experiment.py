"""Lossless BF16 byte packing; reconstructs every original 16-bit pattern.

Keeps sign and mantissa verbatim. Stores exponents as four-bit offsets within
128-element blocks; blocks spanning more than 15 exponents use a full-byte
exception table. This changes storage only, never the represented weights.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from kernels.linear import Projection, bench, reduce


def pack(w):
    if w.numel() > 1048576:
        chunks = []
        offset = 0
        for part in w.flatten().split(1048576):
            low, high, meta, ex = pack(part.view(1, -1))
            meta = torch.where(meta >> 8 != 0, meta + (offset << 8), meta)
            chunks.append((low, high, meta, ex[128:]))
            offset += ex.numel() // 128 - 1
        result = [torch.cat([chunk[i] for chunk in chunks]) for i in range(3)]
        result.append(torch.cat([torch.zeros(128, device=w.device, dtype=torch.uint8)] + [c[3] for c in chunks]))
        return tuple(result)
    bits = w.contiguous().view(torch.int16).flatten().to(torch.int32) & 65535
    count = bits.numel()
    bits = F.pad(bits, (0, (-count) % 128))
    exponent = ((bits >> 7) & 255).view(-1, 128)
    base = exponent.amin(1)
    bad = exponent.amax(1) - base > 15
    index = bad.to(torch.int32).cumsum(0).to(torch.int32)
    meta = torch.where(bad, index << 8, base).to(torch.int32)
    exceptional = torch.cat((torch.zeros(1, 128, device=w.device, dtype=torch.uint8),
                             exponent[bad].to(torch.uint8)), 0).flatten()
    delta = ((exponent - base[:, None]) & 15).flatten()
    high = (delta[::2] | (delta[1::2] << 4)).to(torch.uint8)
    low = ((bits & 127) | ((bits >> 8) & 128)).to(torch.uint8)
    return low, high, meta, exceptional


def unpack(packed, shape):
    low, high, meta, exceptional = packed
    idx = torch.arange(low.numel(), device=low.device)
    m = meta[idx // 128]
    delta = (high[idx // 2].int() >> ((idx % 2) * 4)) & 15
    exponent = torch.where(m >> 8 != 0,
                           exceptional[(m >> 8) * 128 + idx % 128].int(),
                           (m & 255) + delta)
    bits = (low.int() & 127) | ((low.int() & 128) << 8) | (exponent << 7)
    return bits[:shape[0]*shape[1]].to(torch.int16).view(torch.bfloat16).view(shape)


@triton.jit
def load_bf16(LOW, HIGH, META, EX, off, mask):
    low = tl.load(LOW + off, mask, 0).to(tl.int32)
    high = tl.load(HIGH + off // 2, mask, 0).to(tl.int32)
    meta = tl.load(META + off // 128, mask, 0)
    exceptional = meta >> 8
    full = tl.load(EX + exceptional * 128 + off % 128, mask & (exceptional != 0), 0).to(tl.int32)
    exp = tl.where(exceptional != 0, full, (meta & 255) + ((high >> ((off % 2) * 4)) & 15))
    bits = (low & 127) | ((low & 128) << 8) | (exp << 7)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def packed_gemm(X, LOW, HIGH, META, EX, OUT,
                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    ni, mi, si = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m, n = mi * 16 + tl.arange(0,16), ni * BN + tl.arange(0,BN)
    kk = si * BK + tl.arange(0,BK)
    acc = tl.zeros((16,BN), tl.float32)
    for i in range(tl.cdiv(K, BK*SPLIT)):
        k = kk + i*BK*SPLIT
        x = tl.load(X + m[:,None]*K + k[None,:], (m[:,None]<M)&(k[None,:]<K),0)
        w = load_bf16(LOW,HIGH,META,EX,n[None,:]*K+k[:,None],(n[None,:]<N)&(k[:,None]<K))
        acc += tl.dot(x,w)
    tl.store(OUT+(si*M+m[:,None])*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))


@triton.jit
def packed_gemv(X, LOW, HIGH, META, EX, OUT,
                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    ni,m,si=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    n=ni*BN+tl.arange(0,BN)
    kk=si*BK+tl.arange(0,BK)
    acc=tl.zeros((BN,BK),tl.float32)
    for i in range(tl.cdiv(K,BK*SPLIT)):
        k=kk+i*BK*SPLIT
        x=tl.load(X+m*K+k,k<K,0).to(tl.float32)
        w=load_bf16(LOW,HIGH,META,EX,n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K)).to(tl.float32)
        acc += w*x[None,:]
    tl.store(OUT+(si*M+m)*N+n,tl.sum(acc,1),n<N)


class PackedProjection(Projection):
    def __call__(self,x,packed):
        kind,bn,bk,split=self.config
        m,n,k=self.m,self.n,self.k
        if kind=="gemv":
            packed_gemv[(triton.cdiv(n,bn),m,split)](x,*packed,self.part,m,n,k,bn,bk,split,num_warps=4)
        else:
            packed_gemm[(triton.cdiv(n,bn),triton.cdiv(m,16),split)](x,*packed,self.part,m,n,k,bn,bk,split,num_warps=4,num_stages=1)
        if split>1:
            reduce[(triton.cdiv(m*n,256),)](self.part,self.out,m*n,split,256)
        return self.out


@torch.inference_mode()
def main():
    torch.manual_seed(123)
    # Every possible BF16 bit pattern, including signed zeros and NaN payloads.
    bits = torch.arange(65536,device="cuda").to(torch.int16).reshape(256,256)
    for values in (bits, bits.flatten()[torch.randperm(65536,device="cuda")].view(256,256)):
        restored = unpack(pack(values.view(torch.bfloat16)), values.shape)
        assert torch.equal(restored.view(torch.int16),values)
    print("all 65536 BF16 bit patterns round-trip exactly",flush=True)
    for m in (1,4,16):
        for n,k in ((6144,2560),(19456,2560),(2560,9728),(151936,2560)):
            x=torch.randn(m,k,device="cuda",dtype=torch.bfloat16)
            w=(torch.randn(n,k,device="cuda",dtype=torch.float32)*.02).bfloat16()
            packed=pack(w)
            for offset in range(0, w.numel(), 1048576):
                count=min(1048576,w.numel()-offset)
                lp,hp,mp,ep=packed
                restored=unpack((lp[offset:offset+count],hp[offset//2:(offset+count)//2],mp[offset//128:(offset+count)//128],ep),(1,count))
                assert torch.equal(restored.view(torch.int16).flatten(),w.flatten()[offset:offset+count].view(torch.int16))
            ratio=sum(t.numel()*t.element_size() for t in packed)/(w.numel()*2)
            copies=min(8,max(1,triton.cdiv(64*1024*1024,w.numel()*2)+1))
            weights=[w]+[w.clone() for _ in range(copies-1)]
            packs=[packed]+[tuple(t.clone() for t in packed) for _ in range(copies-1)]
            expected=F.linear(x,w)
            baseline=bench(F.linear,x,weights)
            results=[(baseline,"torch")]
            configs=[("gemm",32,64,1),("gemm",64,64,1),("gemm",64,64,4),("gemm",64,128,4),("gemm",64,64,8)]
            if m==1:
                configs += [("gemv",4,512,1),("gemv",4,1024,1),("gemv",8,512,1),("gemv",4,512,4)]
            for config in configs:
                old=Projection(m,n,k,config)
                candidate=PackedProjection(m,n,k,config)
                torch.testing.assert_close(candidate(x,packed),expected,rtol=.02,atol=.02)
                results.append((bench(old,x,weights),("plain",config)))
                results.append((bench(candidate,x,packs),("packed",config)))
            print((m,n,k),"bytes",round(ratio,3),"torch us",round(baseline,2),"best",sorted(results,key=lambda p:p[0])[:4],flush=True)
            del weights,packs,packed,w,expected


if __name__ == "__main__":
    main()
