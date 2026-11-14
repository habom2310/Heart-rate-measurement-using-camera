import React, { useEffect, useRef } from 'react';
import { Activity } from 'lucide-react';

interface WaveformChartProps {
  filteredPulse: number[];
  bpm: number;
}

export const WaveformChart: React.FC<WaveformChartProps> = ({ filteredPulse, bpm }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Maintain rolling smooth display buffer for silky 60fps rendering
  const displayBufferRef = useRef<number[]>([]);

  useEffect(() => {
    if (filteredPulse && filteredPulse.length > 0) {
      displayBufferRef.current = [...filteredPulse];
    }
  }, [filteredPulse]);

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

    let scanOffset = 0;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;
      const dpr = window.devicePixelRatio || 1;

      // 1. Clear background (Deep Matte Obsidian)
      ctx.fillStyle = '#0d0e11';
      ctx.fillRect(0, 0, width, height);

      // 2. Draw precision technical grid lines
      ctx.lineWidth = 1 * dpr;
      ctx.strokeStyle = 'rgba(46, 50, 59, 0.45)';
      const gridSpacingX = 40 * dpr;
      const gridSpacingY = 30 * dpr;

      ctx.beginPath();
      for (let x = 0; x < width; x += gridSpacingX) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      for (let y = 0; y < height; y += gridSpacingY) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();

      // Center reference zero-line
      const midY = height / 2;
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(75, 85, 99, 0.6)';
      ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.moveTo(0, midY);
      ctx.lineTo(width, midY);
      ctx.stroke();
      ctx.setLineDash([]);

      const pulse = displayBufferRef.current;

      if (pulse && pulse.length > 3) {
        // Dynamic autoscaling
        let minVal = Infinity;
        let maxVal = -Infinity;
        for (let i = 0; i < pulse.length; i++) {
          if (pulse[i] < minVal) minVal = pulse[i];
          if (pulse[i] > maxVal) maxVal = pulse[i];
        }

        const range = Math.max(0.001, maxVal - minVal);
        const margin = 20 * dpr;
        const availableHeight = height - margin * 2;

        const points: { x: number; y: number }[] = [];
        const stepX = width / Math.max(1, pulse.length - 1);

        for (let i = 0; i < pulse.length; i++) {
          const normY = (pulse[i] - minVal) / range;
          const y = height - margin - normY * availableHeight;
          const x = i * stepX;
          points.push({ x, y });
        }

        // Draw Arterial Crimson blood pulse trace
        ctx.save();
        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
          if (i === 0) ctx.moveTo(points[i].x, points[i].y);
          else {
            const xc = (points[i].x + points[i - 1].x) / 2;
            const yc = (points[i].y + points[i - 1].y) / 2;
            ctx.quadraticCurveTo(points[i - 1].x, points[i - 1].y, xc, yc);
          }
        }
        ctx.strokeStyle = '#e11d48'; // Arterial Crimson
        ctx.shadowColor = 'rgba(225, 29, 72, 0.35)';
        ctx.shadowBlur = 6 * dpr;
        ctx.lineWidth = 2.2 * dpr;
        ctx.stroke();

        // Second pass: Coral highlight core
        ctx.shadowBlur = 0;
        ctx.strokeStyle = '#fb7185';
        ctx.lineWidth = 1.4 * dpr;
        ctx.stroke();

        // Area gradient fill below pulse curve
        ctx.lineTo(width, height);
        ctx.lineTo(0, height);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, midY, 0, height);
        grad.addColorStop(0, 'rgba(225, 29, 72, 0.12)');
        grad.addColorStop(1, 'rgba(225, 29, 72, 0.0)');
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.restore();

        // Leading pulse indicator dot
        const latestPoint = points[points.length - 1];
        if (latestPoint) {
          ctx.save();
          ctx.beginPath();
          ctx.arc(latestPoint.x, latestPoint.y, 3.5 * dpr, 0, Math.PI * 2);
          ctx.fillStyle = '#fb7185';
          ctx.shadowColor = '#e11d48';
          ctx.shadowBlur = 6 * dpr;
          ctx.fill();
          ctx.restore();
        }
      } else {
        // Flatline during initial acquisition
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(225, 29, 72, 0.25)';
        ctx.lineWidth = 1.5 * dpr;
        ctx.moveTo(0, midY);
        ctx.lineTo(width, midY);
        ctx.stroke();

        ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
        ctx.font = `${Math.round(11 * dpr)}px monospace`;
        ctx.textAlign = 'center';
        ctx.fillText('ACQUIRING CARDIAC TELEMETRY...', width / 2, midY - 14 * dpr);
      }

      // Oscilloscope sweep effect
      scanOffset = (scanOffset + 1.5 * dpr) % width;
      const scanGrad = ctx.createLinearGradient(scanOffset - 30 * dpr, 0, scanOffset, 0);
      scanGrad.addColorStop(0, 'transparent');
      scanGrad.addColorStop(1, 'rgba(225, 29, 72, 0.06)');
      ctx.fillStyle = scanGrad;
      ctx.fillRect(Math.max(0, scanOffset - 30 * dpr), 0, 30 * dpr, height);

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return (
    <div className="bg-[#18191e] border border-[#262932] rounded-xl p-4 shadow-md flex flex-col justify-between">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-rose-500 shrink-0" />
          <span className="text-xs font-semibold tracking-wider text-[#f4f4f6] uppercase font-mono">
            PPG Pulse Waveform
          </span>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-rose-950/40 text-rose-300 border border-rose-900/50">
            0.75 - 3.5 Hz
          </span>
        </div>
        <div className="text-[11px] font-mono text-[#9ca3af] flex items-center gap-2.5">
          <span className="tabular-nums">
            {bpm > 0 ? `Locked: ${bpm.toFixed(1)} BPM` : 'Seeking signal...'}
          </span>
          <span
            className={`w-2 h-2 rounded-full transition-colors ${
              bpm > 0
                ? 'bg-rose-500 shadow-[0_0_8px_rgba(225,29,72,0.6)]'
                : 'bg-zinc-600'
            }`}
          />
        </div>
      </div>

      <div className="relative w-full h-40 sm:h-48 overflow-hidden rounded-lg border border-[#262932] bg-[#0d0e11]">
        <canvas
          ref={canvasRef}
          className="w-full h-full block"
        />
        <div className="absolute top-2 left-2 text-[10px] font-mono text-[#6b7280] pointer-events-none select-none">
          CH-1: ZERO-PHASE BVP [AC]
        </div>
        <div className="absolute bottom-2 right-2 text-[10px] font-mono text-rose-400/80 pointer-events-none select-none">
          60 FPS OSCILLOSCOPE
        </div>
      </div>
    </div>
  );
};
