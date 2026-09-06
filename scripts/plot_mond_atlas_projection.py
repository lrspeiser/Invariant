"""Standalone scientific PNG from the numerical source-image arrays (Pillow)."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from mond_atlas_common import ROOT


def make_plot(output):
    regular='C:/Windows/Fonts/segoeui.ttf';bold='C:/Windows/Fonts/segoeuib.ttf'
    font=lambda size,strong=False:ImageFont.truetype(bold if strong else regular,size)
    canvas=Image.new('RGB',(1800,1260),'#f5f7fa');draw=ImageDraw.Draw(canvas)
    draw.text((65,34),'Does the 3D source reproduce the measured image?',font=font(39,True),fill='#10283b')
    draw.text((65,88),'NGC2903 · source-image diagnostic · no rotation velocities used',font=font(25),fill='#3c5568')
    path=ROOT/'work/private/mond-atlas-projection-001/stellar_luminosity-h0p4.npz'
    with np.load(path) as f:
        axis=f['axis'];observed=f['source_mean'];naive=f['original_projected_surface'];repaired=f['projected_surface'];valid=f['evaluation_weight']>0
    use=np.where(np.abs(axis)<=8)[0];cut=np.ix_(use,use);mask=valid[cut]
    observed,naive,repaired=[a[cut] for a in (observed,naive,repaired)]
    vmax=float(np.ceil(np.nanmax(observed[mask])/1000)*1000)
    anchors=np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],float)
    def colors(value):
        t=np.clip(np.log1p(np.maximum(value,0))/np.log1p(vmax),0,1)
        return np.stack([np.interp(t,np.linspace(0,1,5),anchors[:,c]) for c in range(3)],axis=-1).astype('uint8')
    titles=['Measured projected light','Earlier 0.4 kpc thick model','Refitted 0.4 kpc thick model']
    captions=['Reference image','22.9% image mismatch','8.4% image mismatch']
    panel=455;top=181;lefts=[88,648,1208];extent=8.125
    for left,a,title,caption in zip(lefts,(observed,naive,repaired),titles,captions):
        draw.text((left,143),title,font=font(23,True),fill='#10283b')
        rgb=colors(np.where(np.isfinite(a),a,0));rgb[~mask]=[206,212,218]
        tile=Image.fromarray(rgb.transpose(1,0,2)[::-1]).resize((panel,panel),Image.Resampling.NEAREST)
        canvas.paste(tile,(left,top));draw.rectangle((left,top,left+panel,top+panel),outline='#334c60',width=2)
        for t in (-8,-4,0,4,8):
            px=left+(t+extent)/(2*extent)*panel;py=top+panel-(t+extent)/(2*extent)*panel
            draw.line((px,top+panel,px,top+panel+7),fill='#334c60',width=2)
            draw.text((px-12,top+panel+9),str(t),font=font(18),fill='#334c60')
            draw.line((left-7,py,left,py),fill='#334c60',width=2)
            draw.text((left-38,py-11),str(t),font=font(18),fill='#334c60')
        draw.text((left+86,top+panel+36),'Major-axis position (kpc)',font=font(20),fill='#334c60')
        draw.text((left,top+panel+69),caption,font=font(24,True),fill='#10283b')
    draw.text((68,744),'Vertical axes: stretched minor position (kpc); gray: excluded pixels.',font=font(21),fill='#3c5568')
    draw.text((68,776),'Shared log scale. Errors use the full 0–15 kpc region.',font=font(21),fill='#3c5568')
    # Shared color legend, aligned under all source panels.
    legend=np.tile(colors(np.expm1(np.linspace(0,np.log1p(vmax),550)))[None,:,:],(20,1,1))
    canvas.paste(Image.fromarray(legend),(1160,748))
    draw.text((1160,776),'0',font=font(17),fill='#3c5568')
    draw.text((1540,776),f'{vmax:g} Lsun/pc²',font=font(17),fill='#3c5568')
    data=list(csv.DictReader((ROOT/'work/gravity-first-principles/mond-atlas-projection-001/source-closure.csv').open()))
    origin=(100,1135);w=840;h=285
    draw.text((70,816),'Mismatch grows when thickness is added without reprojection',font=font(25,True),fill='#10283b')
    for value in (0,10,20,30,40,50):
        y=origin[1]-value/50*h;draw.line((origin[0],y,origin[0]+w,y),fill='#d3dbe2',width=1)
        draw.text((origin[0]-43,y-12),str(value),font=font(18),fill='#334c60')
    for value in (0,.1,.2,.4,.8):
        x=origin[0]+value/.8*w;draw.line((x,origin[1],x,origin[1]+7),fill='#334c60',width=2)
        draw.text((x-13,origin[1]+11),str(value),font=font(18),fill='#334c60')
    draw.text((100,1169),'Assumed exponential height (kpc)',font=font(21),fill='#334c60')
    draw.text((73,1207),'Y axis: source-image RMS mismatch (%)   ·   dashed: earlier lift   ·   solid: refitted source',font=font(19),fill='#334c60')
    colors_by_component={'stellar_luminosity':'#a43c20','atomic_helium':'#006e9d','co21':'#7351a4'}
    for component,color in colors_by_component.items():
        selected=[r for r in data if r['component']==component]
        for key,dashed in (('unchanged_lift_relative_image_rms',True),('refitted_source_relative_image_rms',False)):
            points=[(origin[0]+float(r['height_kpc'])/.8*w,origin[1]-100*float(r[key])/50*h) for r in selected]
            if not dashed:draw.line(points,fill=color,width=4)
            else:
                for p,q in zip(points,points[1:]):
                    distance=np.linalg.norm(np.array(q)-p)
                    for start in np.arange(0,distance,15):
                        a=np.array(p)+(np.array(q)-p)*start/distance;b=np.array(p)+(np.array(q)-p)*min(start+8,distance)/distance
                        draw.line((tuple(a),tuple(b)),fill=color,width=3)
            if not dashed:
                for x,y in points:draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
    x0=1040;y0=845
    for text,color in [('Stars','#a43c20'),('Atomic gas','#006e9d'),('CO tracer','#7351a4')]:
        draw.line((x0,y0+14,x0+36,y0+14),fill=color,width=4);draw.text((x0+47,y0),text,font=font(23),fill='#10283b');y0+=42
    draw.text((1040,992),'Different depths can fit the same image',font=font(24,True),fill='#10283b')
    for k,text in enumerate(['0.1 kpc stellar layer: 0.41% mismatch',
                             '0.2 kpc stellar layer: 3.58% mismatch',
                             '25% thin + 75% thick light: 3.63%',
                             'These are source fits, not measured depths.',
                             'Weights are coverage, not a noise likelihood.']):
        draw.text((1040,1035+k*34),text,font=font(21),fill='#3c5568')
    output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output)
    return dict(path=str(output),width=1800,height=1260,brightness_max_lsun_pc2=vmax)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(make_plot(a.output.resolve()))
