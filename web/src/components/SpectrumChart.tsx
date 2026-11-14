import React, { useEffect, useRef } from 'react';
import { BarChart3 } from 'lucide-react';

interface SpectrumChartProps {
  frequencies: number[]; // BPM values
  psd: number[];
  peakBpm: number;
}

export const SpectrumChart: React.FC<SpectrumChartProps> = ({
  frequencies,
  psd,
  peakBpm,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Handle HiDPI Retina and responsive dynamic canvas resizing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const updateCanvasSize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const targetWidth = Math.round(rect.width * dpr);
      const targetHeight = Math.round(rect.height * dpr);

      if (targetWidth > 0 && targetHeight > 0) {
        if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
          canvas.width = targetWidth;
          canvas.height = targetHeight;
        }
      }
    };

    updateCanvasSize();
    const ro = new ResizeObserver(updateCanvasSize);
    ro.observe(canvas);

    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const dpr = window.devicePixelRatio || 1;

    // Clear background (Deep Matte Obsidian)
    ctx.fillStyle = '#0d0e11';
    ctx.fillRect(0, 0, width, height);

    // Padding for axes
    const padLeft = 36 * dpr;
    const padRight = 16 * dpr;
    const padBottom = 26 * dpr;
    const padTop = 16 * dpr;

    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Technical grid lines
    ctx.lineWidth = 1 * dpr;
    ctx.strokeStyle = 'rgba(46, 50, 59, 0.45)';
    const bpmTicks = [60, 90, 120, 150, 180, 210];
    const minBpmAxis = 40;
    const maxBpmAxis = 220;

    // Vertical tick lines & labels
    ctx.fillStyle = 'rgba(156, 163, 175, 0.8)';
    ctx.font = `${Math.round(10 * dpr)}px monospace`;
    ctx.textAlign = 'center';

    bpmTicks.forEach((tick) => {
      const x = padLeft + ((tick - minBpmAxis) / (maxBpmAxis - minBpmAxis)) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();

      ctx.fillText(`${tick}`, x, height - 8 * dpr);
    });

    // Horizontal grid
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop + plotH / 2);
    ctx.lineTo(padLeft + plotW, padTop + plotH / 2);
    ctx.moveTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    if (!frequencies || frequencies.length < 2 || !psd || psd.length < 2) {
      ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
      ctx.font = `${Math.round(11 * dpr)}px monospace`;
      ctx.textAlign = 'center';
      ctx.fillText('ACCUMULATING SPECTRAL PSD...', width / 2, height / 2);
      return;
    }

    // Find max PSD for scaling
    let maxPsd = 0;
    for (let i = 0; i < psd.length; i++) {
      if (psd[i] > maxPsd) maxPsd = psd[i];
    }
    if (maxPsd <= 0) maxPsd = 1;

    // Draw spectrum curve
    const points: { x: number; y: number; bpm: number; power: number }[] = [];
    for (let i = 0; i < frequencies.length; i++) {
      const bpmVal = frequencies[i];
      if (bpmVal >= minBpmAxis && bpmVal <= maxBpmAxis) {
        const x = padLeft + ((bpmVal - minBpmAxis) / (maxBpmAxis - minBpmAxis)) * plotW;
        const normY = psd[i] / maxPsd;
        const y = padTop + plotH - normY * (plotH - 6 * dpr);
        points.push({ x, y, bpm: bpmVal, power: psd[i] });
      }
    }

    if (points.length > 1) {
      // Area fill gradient
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(points[0].x, padTop + plotH);
      for (let i = 0; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.lineTo(points[points.length - 1].x, padTop + plotH);
      ctx.closePath();

      const grad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
      grad.addColorStop(0, 'rgba(56, 189, 248, 0.35)'); // Steel Azure
      grad.addColorStop(0.7, 'rgba(14, 116, 144, 0.12)');
      grad.addColorStop(1, 'rgba(13, 14, 17, 0.0)');
      ctx.fillStyle = grad;
      ctx.fill();

      // Stroke outline
      ctx.beginPath();
      for (let i = 0; i < points.length; i++) {
        if (i === 0) ctx.moveTo(points[i].x, points[i].y);
        else ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2 * dpr;
      ctx.stroke();
      ctx.restore();

      // Highlight peak frequency if available
      if (peakBpm >= minBpmAxis && peakBpm <= maxBpmAxis) {
        const peakX = padLeft + ((peakBpm - minBpmAxis) / (maxBpmAxis - minBpmAxis)) * plotW;

        ctx.save();
        // Glowing vertical peak indicator
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.5 * dpr;
        ctx.setLineDash([3 * dpr, 3 * dpr]);
        ctx.beginPath();
        ctx.moveTo(peakX, padTop);
        ctx.lineTo(peakX, padTop + plotH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Peak callout tag
        ctx.fillStyle = '#0369a1';
        ctx.beginPath();
        ctx.roundRect(peakX - 35 * dpr, padTop - 12 * dpr, 70 * dpr, 18 * dpr, 4 * dpr);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.font = `bold ${Math.round(10 * dpr)}px monospace`;
        ctx.textAlign = 'center';
        ctx.fillText(`${peakBpm.toFixed(1)} BPM`, peakX, padTop + 1 * dpr);
        ctx.restore();
      }
    }
  }, [frequencies, psd, peakBpm]);

  return (
    <div className="bg-[#18191e] border border-[#262932] rounded-xl p-4 shadow-md flex flex-col justify-between">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-sky-400 shrink-0" />
          <span className="text-xs font-semibold tracking-wider text-[#f4f4f6] uppercase font-mono">
            Welch Power Spectral Density
          </span>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-sky-950/40 text-sky-300 border border-sky-900/50">
            2048-FFT
          </span>
        </div>
        <div className="text-[11px] font-mono text-sky-300">
          Peak: <span className="font-bold tabular-nums">{peakBpm > 0 ? `${peakBpm.toFixed(1)} BPM` : '--'}</span>
        </div>
      </div>

      <div className="relative w-full h-36 sm:h-44 overflow-hidden rounded-lg border border-[#262932] bg-[#0d0e11]">
        <canvas
          ref={canvasRef}
          className="w-full h-full block"
        />
        <div className="absolute top-2 left-2 text-[10px] font-mono text-[#6b7280] pointer-events-none select-none">
          PSD [dB/Hz]
        </div>
        <div className="absolute bottom-1 right-2 text-[10px] font-mono text-[#6b7280] pointer-events-none select-none">
          FREQUENCY (BPM)
        </div>
      </div>
    </div>
  );
};
