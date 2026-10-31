"""Minimal GANg README animation; all artwork is drawn with matplotlib Agg.

Requires numpy, matplotlib, Pillow and Liberation Sans or Arial.
    python docs/figures/make_gan_gif.py --stills /tmp/gang-gan-stills
Optional: --font-dir PATH to locally available TTFs, --output PATH.
No downloads. Fixed seed and palette; 144 source frames at 12 fps / 12 s.
GIF centisecond timing alternates 80/90 ms to preserve the exact duration.
The training score and descriptor colours are conceptual, not measurements.
"""
from pathlib import Path
import argparse
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image

W,H,FPS=1000,420,12
REAL,SYN,GEN,CRIT,GOOD='#2E4A7D','#E07B39','#3B6FB6','#6B7785','#2A9D8F'
INK,MUTED,LIGHT='#243444','#5B6670','#EDF1F5'
ENDS=np.array([3.,5.5,8.,10.5,12.])
STARTS=np.r_[0.,ENDS[:-1]]
HOLDS=[2.5,5.,7.7,10.2,11.7]
CAPTIONS=['Generator creates a new node','Critic compares it with real nodes',
          'They train against each other until generated nodes look real',
          'New nodes join the MOF similarity graph','']
RNG=np.random.default_rng(3)
PATTERN=np.array([.9,.3,.7,.5,.95,.2,.55,.8])
INITIAL=RNG.uniform(.15,.95,8)
DOTS=RNG.uniform([-19,-12],[19,12],(6,2))
POINTS=np.array([[772,270],[833,289],[902,275],[959,245],[780,214],
                 [854,236],[923,205],[795,150],[862,160],[958,147]],float)
LINKS=[(0,1),(1,2),(2,3),(0,4),(1,5),(2,5),(3,6),(4,5),(5,6),(4,7),(5,8),(6,8),(6,9),(7,8),(8,9)]
NEW=np.array([[821,247],[895,183],[919,248]],float)
# Only schematic screen positions: links are to the nearest real descriptor
# vectors under cosine similarity, never produced by the generator.
features=RNG.random((10,12))
new_features=np.array([features[k]*.9+.1*RNG.random(12) for k in (4,8,2)])
sim=(new_features/np.linalg.norm(new_features,axis=1)[:,None]) @ (features/np.linalg.norm(features,axis=1)[:,None]).T
NEIGHBORS=np.argsort(-sim,axis=1,kind='stable')[:,:3]
fig=plt.figure(figsize=(10,4.2),dpi=100,facecolor='white')
ax=fig.add_axes([0,0,1,1])


def ease(x):
    x=np.clip(x,0,1)
    return float(x*x*(3-2*x))


def phase(t,start,span=.3):
    return ease((t-start)/span)


def txt(x,y,s,px=22,color=INK,**kw):
    return ax.text(x,y,s,fontsize=px*.72,color=color,ha='center',va='center',zorder=10,**kw)


def line(p,q,color='#C4CCD6',lw=1.4,**kw):
    ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=lw,zorder=2,**kw)


def arrow(p,q,color=MUTED,**kw):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=13,lw=1.7,color=color,zorder=3,**kw))


def node(p,color=SYN,r=19,alpha=1):
    ax.add_patch(Circle(p,r,fc=color,ec='white',lw=1.4,alpha=alpha,zorder=7))


def box(x,y,w,h,fc,ec='none',alpha=1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=7',fc=fc,ec=ec,lw=1.3,alpha=alpha))


def descriptor(p,blend=0,real=False,alpha=1):
    x,y=p
    node(p,REAL if real else SYN,alpha=alpha)
    for j in range(8):
        target=np.array(to_rgb(REAL))*PATTERN[j]+(1-PATTERN[j])*np.ones(3)
        initial=np.array(to_rgb(SYN))*INITIAL[j]+(1-INITIAL[j])*np.ones(3)
        color=target if real else initial*(1-blend)+target*blend
        ax.add_patch(Rectangle((x-39+j*10,y-45),8,14,fc=color,ec='none',alpha=alpha,zorder=7))


def scaffold(t):
    # Fixed actor positions throughout generate / judge / learn / insert.
    for p,q in LINKS:
        line(POINTS[p],POINTS[q])
    for p in POINTS:
        node(p,REAL,r=7)
    txt(865,326,'Similarity graph',color=REAL)
    txt(240,326,'Generator G',color=GEN,fontweight='bold')
    txt(540,326,'Critic D',color=CRIT,fontweight='bold')
    txt(667,326,'Real node',color=REAL)
    pulse=1+.07*np.sin(np.pi*phase(t,1.,.6)) if t<1.6 else 1
    gp=np.array([[-44,-50],[-44,50],[44,68],[44,-68]])*pulse+[240,235]
    ax.add_patch(Polygon(gp,fc='#E3ECF8',ec=GEN,lw=2))
    txt(240,235,'G',px=48,color=GEN,fontweight='bold')
    ax.add_patch(Polygon([[493,167],[493,303],[587,285],[587,185]],fc=LIGHT,ec=CRIT,lw=2))
    txt(540,235,'D',px=48,color=CRIT,fontweight='bold')
    arrow((134,235),(191,235))
    arrow((291,235),(363,235),GEN)
    arrow((426,235),(488,235),SYN)
    arrow((642,235),(592,235),REAL)
    # Inputs slide toward G once, then remain as small steady sources.
    move=phase(t,.2,.8) if t<1.3 else 0
    alpha=1-phase(t,.85,.35) if t<1.3 else phase(t,1.3,.3)
    for p in DOTS:
        node(p+[75+move*135,258-move*23],CRIT,r=3.6,alpha=alpha)
    txt(75,294,'noise z',color=MUTED)
    box(31+move*135,174+move*25,88,35,'#FBE3D3',SYN,alpha)
    txt(75+move*135,191+move*25,'target c',px=20,color=SYN,alpha=alpha)


def score(value,opacity=1):
    # Conceptual critic score, not a probability or measured performance.
    line((490,140),(590,140),CRIT,lw=6,alpha=opacity)
    x=490+100*value
    node((x,140),GOOD if value>=.48 else SYN,r=7,alpha=opacity)
    txt(540,110,"can't tell" if value>=.48 else 'generated',px=20,color=CRIT,alpha=opacity)


def draw(t):
    ax.clear();ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')
    stage=min(np.searchsorted(ENDS,t,side='right'),4)
    ax.add_patch(Rectangle((0,0),W,65,fc='#F5F7FA',ec='none'))
    if stage==4:
        txt(500,235,'GANg — GAN for graph-node synthesis',px=39,color=GEN,fontweight='bold')
        return
    scaffold(t)
    ax.text(30,377,['Generate','Judge','Learn','Insert'][stage],fontsize=18,color=GEN,fontweight='bold',va='center')
    txt(500,31,CAPTIONS[stage],px=23)
    if stage==0:
        p=phase(t,1.35,.6)
        descriptor((290+105*p,235),alpha=p)
        descriptor((667,235),real=True)
    elif stage==1:
        p=phase(t,3.3,.8)
        fade=1-phase(t,3.85,.3)
        descriptor((395+105*p,235),alpha=fade)
        descriptor((667-85*p,235),real=True,alpha=fade)
        score(.15,phase(t,4.1,.3))
    elif stage==2:
        local=min(t-5.5,2.05)
        cycle=min(int(local/.65),2)
        u=min((local-cycle*.65)/.65,1)
        blend=min((cycle+ease(u))/3,1)
        # Three visible G -> descriptor -> D passes; a real comparison stays visible.
        x=291+205*ease(u)
        descriptor((395 if local>=1.95 else x,235),blend=blend,
                   alpha=1 if local>=1.95 else 1-phase(u,.82,.18))
        descriptor((667,235),real=True)
        arrow((492,185),(246,172),GOOD,connectionstyle='arc3,rad=-.45',alpha=phase(t,5.5,.3))
        txt(360,90,'feedback',px=20,color=GOOD)
        score(.15+.35*blend)
    elif stage==3:
        descriptor((667,235),real=True)
        for j,dest in enumerate(NEW):
            start=8.+j*.48
            if t<start:
                continue
            p=phase(t,start,.42)
            pos=np.array([395.,235.])*(1-p)+dest*p
            # Route the flight below the actors, clear of labels and captions.
            pos[1]-=135*np.sin(np.pi*p)
            node(pos,SYN,r=19-9*p)
            for k,nb in enumerate(NEIGHBORS[j]):
                q=phase(t,start+.42+k*.10,.16)
                if q>0:
                    line(dest,dest+(POINTS[nb]-dest)*q,SYN,lw=1.5,ls=(0,(3,2)))
        txt(395,290,'Similarity links',px=20,color=GOOD)


def render(t):
    draw(t);fig.canvas.draw()
    for text in ax.texts:
        b=text.get_window_extent(fig.canvas.get_renderer())
        assert 0<=b.x0 and b.x1<=W and 0<=b.y0 and b.y1<=H,(t,text.get_text(),b)
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font-dir',type=Path)
    parser.add_argument('--stills',type=Path)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('gang_gan.gif'))
    args=parser.parse_args()
    if args.font_dir:
        for font in sorted(args.font_dir.glob('*.ttf')):
            font_manager.fontManager.addfont(str(font))
    available={f.name for f in font_manager.fontManager.ttflist}
    family=next((f for f in ['Liberation Sans','Arial'] if f in available),None)
    if family is None:
        parser.error('Install Liberation Sans/Arial or supply local TTFs with --font-dir')
    plt.rcParams['font.family']=[family]
    holds=[render(t) for t in HOLDS]
    if args.stills:
        args.stills.mkdir(parents=True,exist_ok=True)
        for i,im in enumerate(holds,1):
            im.save(args.stills/f'step_{i}.png')
    sample=Image.new('RGB',(W,H*5),'white')
    for i,im in enumerate(holds):sample.paste(im,(0,H*i))
    palette=sample.quantize(colors=64,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE)
    frames=[]
    for n in range(144):
        t=n/FPS
        i=min(np.searchsorted(ENDS,t,side='right'),4)
        # Pause the completed step, then crossfade into the next for 0.3 s.
        im=render(min(t,HOLDS[i]))
        if t-STARTS[i]<.3:
            im=Image.blend(holds[(i-1)%5],im,phase(t,STARTS[i],.3))
        frames.append(im.quantize(palette=palette,dither=Image.Dither.NONE))
    durations=[round((n+1)*100/FPS)*10-round(n*100/FPS)*10 for n in range(144)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True,disposal=1)
    size=args.output.stat().st_size
    with Image.open(args.output) as gif:
        assert gif.size==(W,H) and gif.info['loop']==0
        elapsed=0
        for n in range(gif.n_frames):
            gif.seek(n);elapsed+=gif.info['duration']
        count=gif.n_frames
    assert elapsed==12000 and size<3_000_000
    print(f'{size:,} bytes; 144 rendered frames, {count} encoded; 12 fps; {elapsed/1000:g} s; infinite loop')
    print('SHA256',hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__=='__main__':main()
