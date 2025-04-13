import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
        
class Sudoku4x4RandomPatternDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        self.size = 4
        self.difficulty = 0.4
        
    def __iter__(self):
        return self

    def __next__(self):
        n = self.size
        target = torch.stack([torch.arange(n**2).add(1).roll(i).roll(j*n) for i in range(n) for j in range(n)])
        for i in range(n):
            for r in range(0,n**2,n):
                target[r:r+n,:] = target[torch.randperm(n)+r,:]
            for c in range(0,n**2,n):
                target[:,c:c+n] = target[:,torch.randperm(n)+c]
    
        challenge = target.clone()
        challenge[torch.rand_like(challenge.float()) < self.difficulty] = 0
        pchallenge = torch.zeros_like(challenge).repeat(4,4)[:-1,:-1] #F.interpolate(buffer[None,None].float(), scale_factor=self.scale, mode="nearest")[0,0].long()
        ptarget = torch.zeros_like(target).repeat(4,4)[:-1,:-1] #F.interpolate(target[None,None].float(), scale_factor=self.scale, mode="nearest")[0,0].long()
        
        patterns = F.pad(torch.rand(16,3,3).gt(0.5) * torch.randint(1,11,(16,1,1)),(0,0,0,0,1,0))
 
        for row in range(self.size**2):
            for col in range(self.size**2):
                pchallenge[row*4:row*4+3,col*4:col*4+3] = patterns[challenge[row,col].item()]
                ptarget[row*4:row*4+3,col*4:col*4+3] = patterns[target[row,col].item()]

        return pchallenge[None], ptarget[None] 

      
class RestorePatternDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        self.size = (32,32)
        self.min_size = (6,6)
        self.max_size = (30,30)
        self.min_pattern_size = (1,1)
        self.max_pattern_size = (4,4)
        self.occlusion = ["solid", "random"]
        self.strength = 0.75
        
    def __iter__(self):
        return self

    def __next__(self):
        h = torch.randint(self.min_size[0], self.max_size[0]+1, (1,)).item()
        w = torch.randint(self.min_size[1], self.max_size[1]+1, (1,)).item()
        ph = torch.randint(self.min_pattern_size[0], self.max_pattern_size[0]+1, (1,)).item()
        pw = torch.randint(self.min_pattern_size[1], self.max_pattern_size[1]+1, (1,)).item()
        pattern = torch.randint(10, (ph,pw))+1
        challenge = pattern.repeat((h+ph-1)//ph, (w+pw-1)//pw)[:h,:w]
        target = challenge.clone()
        
        ph = torch.randint(2, h-2, (1,)).item()
        pw = torch.randint(2, w-2, (1,)).item()
        dy = torch.randint(h-ph,(1,))
        dx = torch.randint(w-pw,(1,))
        pattern = (torch.rand(ph,pw) < self.strength)

        occlusion = self.occlusion[torch.randint(len(self.occlusion), (1,)).item()]
        if occlusion == "solid":
            color = torch.randint(10, (1,)).item() + 1
            challenge[dy:dy+ph, dx:dx+pw][pattern] = color
        elif occlusion == "random":
            color = torch.randint(10, (ph,pw)) + 1
            challenge[dy:dy+ph, dx:dx+pw][pattern] = color[pattern]
        else:
            assert False, "invalid occlusion"

        py,px = (self.size[0]-h),(self.size[1]-w)
        challenge = F.pad(challenge,(px//2, px-px//2, py//2, py-py//2))
        target = F.pad(target,(px//2, px-px//2, py//2, py-py//2))
        
        return challenge[None], target[None]

class MatchPatternDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        self.size = (32,32)
        self.occlusion = ["solid", "random"]
        self.rotate = True

    def __iter__(self):
        return self

    def __next__(self):
        h,w = self.size
        ph,pw = 3,3
        rot = torch.randint(4,()).item() if self.rotate else 0
        colors = torch.randint(1,10, (ph,pw))
        challenge = torch.zeros(h,w, dtype=torch.long)
        pidx = torch.randint(4, (1,)).item()
        
        while True:
            patterns = torch.rand(4,ph,pw) > 0.5
            if any(patterns[pidx].eq( patterns[pidx].rot90(r, (-1,-2))).all() for r in range(1,4)):
                continue
            if any(p.eq(q.rot90(r,(-1,-2))).all() for i,p in enumerate(patterns) for q in patterns[i+1:] for r in range(4)):
                continue
            break
        colors_x2 = F.interpolate(colors[None,None].float(), scale_factor=3, mode="nearest")[0,0].long()
        pattern_x2 = F.interpolate(patterns[[pidx],None].float(), scale_factor=3, mode="nearest")[0,0].long()
        pattern_x2 = pattern_x2.rot90(rot, (-1,-2))
        colors_x2 = colors_x2.rot90(rot, (-1,-2))
        ph,pw = ph*3, pw*3
        challenge[(h-ph)//2:(h-ph)//2+ph, (w//2-pw)//2:(w//2-pw)//2+pw] = pattern_x2 * colors_x2
        target = challenge.clone()

        for j,pattern in enumerate(patterns):
            pattern_x2 = F.interpolate(pattern[None,None].float(), scale_factor=2, mode="nearest")[0,0].long()
            colors_x2 = F.interpolate(colors[None,None].float(), scale_factor=2, mode="nearest")[0,0].long()
            h,w = 6,6
            r,c = (j % 2)*16, (j // 2) * 8
            oy,ox = 4,1
            challenge[oy+r:oy+r+h, 16+ox+c:16+ox+c+w] = pattern_x2 * 10 
            
            if pidx == j:
                target[oy+r:oy+r+h, 16+ox+c:16+ox+c+w] = pattern_x2 * colors_x2             
            else:
                target[oy+r:oy+r+h, 16+ox+c:16+ox+c+w] = pattern_x2 * 10             
        
        return challenge[None], target[None]


class SquaresDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        self.size = (32,32)
        self.min_lines = 1
        self.max_lines = 4
        
    def __iter__(self):
        return self
        
    def __next__(self):
        h,w = self.size 
        target = torch.zeros(h,w, dtype=torch.long)
        challenge = torch.zeros(h,w, dtype=torch.long)
        num_lines = torch.randint(self.min_lines,self.max_lines+1,()).item()
        
        line_xx = torch.randperm(w)[:num_lines*2].view(-1,2)
        line_yy = torch.randperm(h)[:num_lines*2].view(-1,2)
        colors = torch.randperm(10).add(1)[:num_lines].sort(0)[0]
        
        for (x0,x1),(y0,y1),color in zip(line_xx, line_yy, colors):
            challenge[y0,x0] = color
            challenge[y1,x1] = color
            x0,x1 = sorted([x0,x1])
            y0,y1 = sorted([y0,y1])
            target[y0:y1+1,x0] = color
            target[y0:y1+1,x1] = color
            target[y0,x0:x1+1] = color
            target[y1,x0:x1+1] = color
            
        return challenge[None], target[None]


class LinesDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        self.size = (32,32)
        self.min_lines = 1
        self.max_lines = 4
        
    def __iter__(self):
        return self
        
    def __next__(self):
        h,w = self.size 
        target = torch.zeros(h,w, dtype=torch.long)
        challenge = torch.zeros(h,w, dtype=torch.long)
        num_lines = torch.randint(self.min_lines,self.max_lines+1,()).item()

        line_dx = torch.randint(1,5,(num_lines,)) * torch.randn(num_lines).sign().long()
        line_dy = torch.randint(3,(num_lines,)) - 1
        line_len = torch.randint(min(w,h), (num_lines,))
        colors = torch.randperm(10).add(1)[:num_lines].sort(0)[0]
        
        for dx,dy,ll,color in zip(line_dx, line_dy, line_len, colors):
            if torch.rand(()) > 0.5:
                dx,dy = dy,dx
            ll = max(1,ll//max(abs(dx),abs(dy)))
            lw,lh = (abs(dx * ll).item(), abs(dy * ll).item())
            sx,sy = torch.randint(w-lw,()).item(), torch.randint(h-lh,()).item()
            if dx < 0: sx = w-sx-1
            if dy < 0: sy = h-sy-1
                
            for i in range(ll):
                x0,y0 = min(max(0,sx+dx*i),self.size[1]-1), min(max(0,sy+dy*i),self.size[0]-1)
                x1,y1 = min(max(0,sx+dx*(i+1)),self.size[1]-1), min(max(0,sy+dy*(i+1)),self.size[0]-1)
                x0,x1 = sorted([x0,x1])
                y0,y1 = sorted([y0,y1])
                if x0 == x1: x1 = x0 + 1
                if y0 == y1: y1 = y0 + 1
                target[y0:y1,x0:x1] = color

            x0,x1 = sx,min(max(0,sx+dx*ll),self.size[1]-1)
            y0,y1 = sy,min(max(0,sy+dy*ll),self.size[0]-1)
            if x0 != x1: x1 -= 1
            if y0 != y1: y1 -= 1
            challenge[y0,x0] = color
            challenge[y1,x1] = color
            
            
        return challenge[None], target[None]