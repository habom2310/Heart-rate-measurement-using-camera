import React from 'react';
import { Activity, AlertTriangle, CheckCircle2, Clock, Cpu, Heart, ShieldAlert } from 'lucide-react';
import type { HeartRateResult } from '../types/rppg';

interface VitalCardsProps {
  result: HeartRateResult | null;
  bufferFill: number;
  bufferCapacity: number;
  fps: number;
}

export const VitalCards: React.FC<VitalCardsProps> = ({
  result,
  bufferFill,
  bufferCapacity,
  fps,
}) => {
  const bpm = result && result.bpm > 0 ? result.bpm.toFixed(1) : '--';
  const snr = result && result.snr > -25 ? `${result.snr.toFixed(1)} dB` : '--';
  const quality = result?.quality || 'Low';
  const motionWarning = result?.motionDetected ?? false;

  const rmssd = result?.hrv?.rmssd ? `${result.hrv.rmssd.toFixed(1)} ms` : '--';
  const sdnn = result?.hrv?.sdnn ? `${result.hrv.sdnn.toFixed(1)} ms` : '--';
  const pnn50 = result?.hrv?.pnn50 !== undefined ? `${result.hrv.pnn50.toFixed(1)}%` : '--';

  const bufferPercent = Math.min(100, Math.round((bufferFill / bufferCapacity) * 100));

  // Dynamic heart pulse rate calculation for CSS animation
  const pulseDuration = result && result.bpm > 40 ? `${(60.0 / result.bpm).toFixed(2)}s` : '1s';

  const getQualityBadge = () => {
    switch (quality) {
      case 'Good':
        return 'text-emerald-300 bg-emerald-950/40 border-emerald-800/50';
      case 'Fair':
        return 'text-amber-300 bg-amber-950/40 border-amber-800/50';
      case 'Low':
      default:
        return 'text-rose-300 bg-rose-950/40 border-rose-800/50';
    }
  };

  return (
    <div className="grid grid-cols-2 gap-2.5 sm:gap-3 w-full">
      {/* 1. Primary Heart Rate Card */}
      <div className="bg-[#18191e] border border-[#262932] rounded-xl p-3 sm:p-4 shadow-sm relative overflow-hidden flex flex-col justify-between min-w-0">
        <div className="flex items-center justify-between gap-1.5">
          <span className="text-xs font-semibold tracking-wider text-[#9ca3af] uppercase flex items-center gap-1.5 font-mono whitespace-nowrap">
            <Heart
              className={`w-3.5 h-3.5 sm:w-4 sm:h-4 text-rose-500 fill-rose-500 transition-transform shrink-0 ${
                result && result.bpm > 0 ? 'animate-pulse' : ''
              }`}
              style={{ animationDuration: pulseDuration }}
            />
            <span>Pulse</span>
          </span>
          <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-[#202229] text-[#9ca3af] border border-[#2e323b] shrink-0 whitespace-nowrap">
            Live
          </span>
        </div>

        <div className="my-2 flex items-baseline gap-1.5">
          <span className="text-3xl sm:text-4xl font-extrabold font-mono tracking-tight text-white drop-shadow-[0_0_8px_rgba(225,29,72,0.25)] tabular-nums">
            {bpm}
          </span>
          <span className="text-xs sm:text-sm font-medium text-[#9ca3af] font-mono">BPM</span>
        </div>

        <div className="flex items-center justify-between text-[10px] sm:text-xs text-[#9ca3af] pt-1.5 border-t border-[#262932] font-mono gap-1">
          <span className="whitespace-nowrap">60 - 100</span>
          <span className="text-rose-400 whitespace-nowrap">0.1 Res</span>
        </div>
      </div>

      {/* 2. Signal Quality Index (SQI / SNR) Card */}
      <div className="bg-[#18191e] border border-[#262932] rounded-xl p-3 sm:p-4 shadow-sm relative overflow-hidden flex flex-col justify-between min-w-0">
        <div className="flex items-center justify-between gap-1.5">
          <span className="text-xs font-semibold tracking-wider text-[#9ca3af] uppercase flex items-center gap-1.5 font-mono whitespace-nowrap">
            <Activity className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-emerald-400 shrink-0" />
            <span>SQI</span>
          </span>
          <div
            className={`flex items-center gap-1 px-1.5 py-0.5 rounded border text-[10px] font-medium font-mono shrink-0 whitespace-nowrap ${getQualityBadge()}`}
          >
            {quality === 'Good' && <CheckCircle2 className="w-2.5 h-2.5 sm:w-3 sm:h-3" />}
            {quality === 'Fair' && <AlertTriangle className="w-2.5 h-2.5 sm:w-3 sm:h-3" />}
            {quality === 'Low' && <ShieldAlert className="w-2.5 h-2.5 sm:w-3 sm:h-3" />}
            <span>{quality}</span>
          </div>
        </div>

        <div className="my-2 flex items-baseline gap-1.5">
          <span className="text-2xl sm:text-3xl font-bold font-mono tracking-tight text-white tabular-nums">
            {snr}
          </span>
          <span className="text-[10px] sm:text-xs font-mono text-[#9ca3af]">SNR</span>
        </div>

        <div className="flex items-center justify-between text-[10px] sm:text-xs font-mono pt-1.5 border-t border-[#262932] gap-1">
          {motionWarning ? (
            <span
              className="text-amber-400 flex items-center gap-1 font-semibold whitespace-nowrap truncate"
              title={
                result?.motionCompensated
                  ? 'Subtle motion detected - inertial tracking active'
                  : 'Motion artifact detected'
              }
            >
              <AlertTriangle className="w-3 h-3 shrink-0" />
              <span>{result?.motionCompensated ? 'Damped' : 'Motion'}</span>
            </span>
          ) : result?.lightCompensated ? (
            <span
              className="text-sky-400 flex items-center gap-1 font-semibold whitespace-nowrap truncate"
              title="Ambient illumination drift normalized by sliding sub-window filter"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-pulse shrink-0" />
              <span>Light Norm</span>
            </span>
          ) : (
            <span className="text-[#9ca3af] whitespace-nowrap">Stable lock</span>
          )}
          <span className="text-[#9ca3af] shrink-0 whitespace-nowrap" title="Values above 6 dB indicate clinical reliability">
            &gt; 6 dB
          </span>
        </div>
      </div>

      {/* 3. HRV Metrics Card */}
      <div className="bg-[#18191e] border border-[#262932] rounded-xl p-3 sm:p-4 shadow-sm relative overflow-hidden flex flex-col justify-between min-w-0">
        <div className="flex items-center justify-between gap-1.5">
          <span className="text-xs font-semibold tracking-wider text-[#9ca3af] uppercase flex items-center gap-1.5 font-mono whitespace-nowrap">
            <Clock className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-sky-400 shrink-0" />
            <span>HRV</span>
          </span>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#202229] text-sky-300 border border-[#2e323b] shrink-0 whitespace-nowrap">
            ms
          </span>
        </div>

        <div className="my-2 grid grid-cols-2 gap-2">
          <div title="RMSSD: Root Mean Square of Successive Differences" className="min-w-0">
            <div className="text-[10px] text-[#9ca3af] font-mono uppercase whitespace-nowrap">RMSSD</div>
            <div className="text-base sm:text-lg font-bold font-mono text-white tabular-nums truncate">{rmssd}</div>
          </div>
          <div title="SDNN: Standard Deviation of NN intervals" className="min-w-0">
            <div className="text-[10px] text-[#9ca3af] font-mono uppercase whitespace-nowrap">SDNN</div>
            <div className="text-base sm:text-lg font-bold font-mono text-white tabular-nums truncate">{sdnn}</div>
          </div>
        </div>

        <div className="flex items-center justify-between text-[10px] sm:text-xs font-mono text-[#9ca3af] pt-1.5 border-t border-[#262932] gap-1">
          <span className="whitespace-nowrap">pNN50: <span className="text-[#f4f4f6] tabular-nums">{pnn50}</span></span>
          <span className="whitespace-nowrap">IBI: {result?.hrv?.ibis?.length || 0}</span>
        </div>
      </div>

      {/* 4. DSP Buffer & Frame Rate Card */}
      <div className="bg-[#18191e] border border-[#262932] rounded-xl p-3 sm:p-4 shadow-sm relative overflow-hidden flex flex-col justify-between min-w-0">
        <div className="flex items-center justify-between gap-1.5">
          <span className="text-xs font-semibold tracking-wider text-[#9ca3af] uppercase flex items-center gap-1.5 font-mono whitespace-nowrap">
            <Cpu className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-zinc-400 shrink-0" />
            <span>Buffer</span>
          </span>
          <span className="text-[10px] sm:text-xs font-mono font-semibold text-[#f4f4f6] tabular-nums shrink-0 whitespace-nowrap">
            {fps.toFixed(0)} FPS
          </span>
        </div>

        <div className="my-2">
          <div className="flex justify-between items-baseline mb-1 text-[10px] sm:text-xs font-mono gap-1">
            <span className="text-[#9ca3af] whitespace-nowrap">Window</span>
            <span className="text-white tabular-nums whitespace-nowrap">
              {bufferFill}/{bufferCapacity} ({bufferPercent}%)
            </span>
          </div>
          <div className="w-full bg-[#202229] rounded-full h-1.5 sm:h-2 overflow-hidden border border-[#2e323b]">
            <div
              className={`h-full transition-all duration-300 rounded-full ${
                bufferPercent >= 100
                  ? 'bg-rose-500'
                  : 'bg-emerald-500'
              }`}
              style={{ width: `${bufferPercent}%` }}
            />
          </div>
        </div>

        <div className="flex items-center justify-between text-[10px] sm:text-xs font-mono text-[#9ca3af] pt-1.5 border-t border-[#262932] gap-1">
          <span className="whitespace-nowrap">~{(1000 / (fps || 30)).toFixed(0)}ms</span>
          <span className="whitespace-nowrap text-[#9ca3af]">
            {bufferPercent >= 100 ? 'Locked' : 'Buffering'}
          </span>
        </div>
      </div>
    </div>
  );
};
