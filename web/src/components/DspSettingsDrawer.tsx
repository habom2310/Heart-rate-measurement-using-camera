import React from 'react';
import {
  Check,
  Cpu,
  Eye,
  Flame,
  Layers,
  ShieldCheck,
  Sliders,
  Sparkles,
  Tv,
  X,
} from 'lucide-react';
import type { RPPGAlgorithmType, ROIMode } from '../types/rppg';

interface DspSettingsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  algorithm: RPPGAlgorithmType;
  onAlgorithmChange: (alg: RPPGAlgorithmType) => void;
  roiMode: ROIMode;
  onRoiModeChange: (mode: ROIMode) => void;
  bufferCapacity: number;
  onBufferSizeChange: (size: number) => void;
  showMesh: boolean;
  onToggleMesh: () => void;
  showROIs: boolean;
  onToggleROIs: () => void;
  showHUD: boolean;
  onToggleHUD: () => void;
  evmEnabled: boolean;
  onToggleEVM: () => void;
  isDemoMode: boolean;
  onToggleDemoMode: () => void;
}

export const DspSettingsDrawer: React.FC<DspSettingsDrawerProps> = ({
  isOpen,
  onClose,
  algorithm,
  onAlgorithmChange,
  roiMode,
  onRoiModeChange,
  bufferCapacity,
  onBufferSizeChange,
  showMesh,
  onToggleMesh,
  showROIs,
  onToggleROIs,
  showHUD,
  onToggleHUD,
  evmEnabled,
  onToggleEVM,
  isDemoMode,
  onToggleDemoMode,
}) => {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-[#111215]/80 backdrop-blur-sm transition-opacity duration-300"
      role="dialog"
      aria-modal="true"
      aria-labelledby="dsp-drawer-title"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md bg-[#18191e] border-l border-[#262932] h-full overflow-y-auto p-5 sm:p-6 shadow-2xl flex flex-col justify-between"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="space-y-6">
          {/* Drawer Header */}
          <div className="flex items-center justify-between pb-4 border-b border-[#262932]">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-center justify-center">
                <Sliders className="w-4 h-4 text-rose-400" />
              </div>
              <div>
                <h2 id="dsp-drawer-title" className="text-base font-bold text-white font-mono">
                  DSP Parameters
                </h2>
                <p className="text-xs text-zinc-400">
                  Signal Extraction &amp; Telemetry Pipeline
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-[#202229] transition-all min-h-[44px] min-w-[44px] flex items-center justify-center cursor-pointer"
              aria-label="Close settings"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Section 1: Extraction Algorithm */}
          <div className="space-y-2.5">
            <label className="text-xs font-semibold text-zinc-300 uppercase tracking-wider font-mono flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-zinc-400" />
              <span>1. Extraction Algorithm</span>
            </label>
            <div className="space-y-2">
              {[
                {
                  id: 'POS' as RPPGAlgorithmType,
                  name: 'POS (Plane-Orthogonal-to-Skin)',
                  desc: 'Recommended. Filters motion artifacts and specular highlights through temporal projection.',
                  badge: 'Standard',
                },
                {
                  id: 'CHROM' as RPPGAlgorithmType,
                  name: 'CHROM (Chrominance Difference)',
                  desc: 'Clinical benchmark. Eliminates intensity fluctuations via standardized color differences.',
                  badge: 'Clinical',
                },
                {
                  id: 'Green' as RPPGAlgorithmType,
                  name: 'Green Channel Mean',
                  desc: 'Direct hemodynamic capillary absorption. Lowest latency, best for still subjects.',
                  badge: 'Fast',
                },
              ].map((item) => {
                const isSelected = algorithm === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => onAlgorithmChange(item.id)}
                    className={`w-full text-left p-3 rounded-xl border transition-all flex items-start justify-between gap-3 min-h-[52px] cursor-pointer ${
                      isSelected
                        ? 'bg-rose-950/25 border-rose-500/50 text-white'
                        : 'bg-[#141519] border-[#262932] hover:border-[#3f4452] text-zinc-300'
                    }`}
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className={`text-xs font-bold font-mono ${isSelected ? 'text-rose-300' : 'text-white'}`}>
                          {item.name}
                        </span>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#202229] text-zinc-400 border border-[#2e323b]">
                          {item.badge}
                        </span>
                      </div>
                      <p className="text-[11px] text-zinc-400 leading-relaxed">
                        {item.desc}
                      </p>
                    </div>
                    {isSelected && (
                      <div className="w-5 h-5 rounded-full bg-rose-500/20 border border-rose-500/50 flex items-center justify-center shrink-0 mt-0.5">
                        <Check className="w-3 h-3 text-rose-400" />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section 2: Temporal Buffer Window */}
          <div className="space-y-2.5">
            <label className="text-xs font-semibold text-zinc-300 uppercase tracking-wider font-mono flex items-center gap-1.5">
              <span>2. Temporal Analysis Window</span>
            </label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { size: 90, label: '~3s (90f)', sub: 'Fast Response' },
                { size: 150, label: '~5s (150f)', sub: 'Recommended' },
                { size: 300, label: '~10s (300f)', sub: 'High SNR / HRV' },
              ].map((item) => {
                const isSelected = bufferCapacity === item.size;
                return (
                  <button
                    key={item.size}
                    onClick={() => onBufferSizeChange(item.size)}
                    className={`p-2.5 rounded-xl border text-center transition-all min-h-[52px] flex flex-col items-center justify-center cursor-pointer ${
                      isSelected
                        ? 'bg-rose-950/30 border-rose-500/50 text-rose-300'
                        : 'bg-[#141519] border-[#262932] hover:border-[#3f4452] text-zinc-300'
                    }`}
                  >
                    <span className="text-xs font-bold font-mono block">
                      {item.label}
                    </span>
                    <span className="text-[10px] text-zinc-400 font-mono mt-0.5 block">
                      {item.sub}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section 3: ROI Sampling Region */}
          <div className="space-y-2.5">
            <label className="text-xs font-semibold text-zinc-300 uppercase tracking-wider font-mono flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-emerald-400" />
              <span>3. ROI Sampling Region</span>
            </label>
            <div className="space-y-2">
              {[
                {
                  id: 'cheeks' as ROIMode,
                  name: 'Cheeks Only (High Malar Bones)',
                  desc: 'Recommended default. Immune to hair/bangs and beards. Samples high-capillary infraorbital cheekbones.',
                  badge: 'Default',
                },
                {
                  id: 'all' as ROIMode,
                  name: 'Forehead + Cheeks',
                  desc: 'Full upper-facial sampling. Best when forehead is unobstructed by bangs or headwear.',
                  badge: 'Full Face',
                },
                {
                  id: 'forehead' as ROIMode,
                  name: 'Forehead Only',
                  desc: 'Upper-face only. Ideal when wearing medical face masks, respirators, or scarves.',
                  badge: 'Mask Safe',
                },
              ].map((item) => {
                const isSelected = roiMode === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => onRoiModeChange(item.id)}
                    className={`w-full text-left p-3 rounded-xl border transition-all flex items-start justify-between gap-3 min-h-[52px] cursor-pointer ${
                      isSelected
                        ? 'bg-rose-950/25 border-rose-500/50 text-white'
                        : 'bg-[#141519] border-[#262932] hover:border-[#3f4452] text-zinc-300'
                    }`}
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className={`text-xs font-bold font-mono ${isSelected ? 'text-rose-300' : 'text-white'}`}>
                          {item.name}
                        </span>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#202229] text-zinc-400 border border-[#2e323b]">
                          {item.badge}
                        </span>
                      </div>
                      <p className="text-[11px] text-zinc-400 leading-relaxed">
                        {item.desc}
                      </p>
                    </div>
                    {isSelected && (
                      <div className="w-5 h-5 rounded-full bg-rose-500/20 border border-rose-500/50 flex items-center justify-center shrink-0 mt-0.5">
                        <Check className="w-3 h-3 text-rose-400" />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section 4: Visual Overlays */}
          <div className="space-y-2.5">
            <label className="text-xs font-semibold text-zinc-300 uppercase tracking-wider font-mono flex items-center gap-1.5">
              <Tv className="w-3.5 h-3.5 text-zinc-400" />
              <span>4. Telemetry Visual Overlays</span>
            </label>
            <div className="space-y-2">
              {[
                {
                  label: 'Forehead & Cheek ROIs',
                  desc: 'Draw real-time green convex hulls over micro-capillary sampling regions',
                  active: showROIs,
                  toggle: onToggleROIs,
                  icon: Layers,
                  color: 'text-emerald-400',
                },
                {
                  label: 'Facial Mesh Wireframe',
                  desc: 'Display 468-point anatomical mesh contour geometry',
                  active: showMesh,
                  toggle: onToggleMesh,
                  icon: Eye,
                  color: 'text-zinc-400',
                },
                {
                  label: 'Medical HUD Reticle',
                  desc: 'Corner alignment brackets and extracted mean RGB readouts',
                  active: showHUD,
                  toggle: onToggleHUD,
                  icon: Tv,
                  color: 'text-sky-400',
                },
              ].map((item) => (
                <button
                  key={item.label}
                  onClick={item.toggle}
                  className="w-full p-3 rounded-xl bg-[#141519] border border-[#262932] hover:border-[#3f4452] flex items-center justify-between gap-3 text-left transition-all min-h-[48px] cursor-pointer"
                >
                  <div className="flex items-center gap-3">
                    <item.icon className={`w-4 h-4 ${item.color} shrink-0`} />
                    <div>
                      <span className="text-xs font-semibold text-white block">
                        {item.label}
                      </span>
                      <p className="text-[11px] text-zinc-400">
                        {item.desc}
                      </p>
                    </div>
                  </div>
                  <div
                    className={`w-9 h-5 rounded-full transition-colors relative shrink-0 ${
                      item.active ? 'bg-rose-600' : 'bg-[#202229] border border-[#2e323b]'
                    }`}
                  >
                    <div
                      className={`w-4 h-4 rounded-full bg-white transition-transform absolute top-0.5 ${
                        item.active ? 'left-4' : 'left-0.5'
                      }`}
                    />
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Section 5: Experimental & Testing */}
          <div className="space-y-2.5">
            <label className="text-xs font-semibold text-zinc-300 uppercase tracking-wider font-mono flex items-center gap-1.5">
              <span>5. Experimental &amp; Simulation</span>
            </label>
            <div className="space-y-2">
              <button
                onClick={onToggleEVM}
                className={`w-full p-3 rounded-xl border flex items-center justify-between gap-3 text-left transition-all min-h-[48px] cursor-pointer ${
                  evmEnabled
                    ? 'bg-rose-950/30 border-rose-500/50'
                    : 'bg-[#141519] border-[#262932] hover:border-[#3f4452]'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Flame className={`w-4 h-4 ${evmEnabled ? 'text-rose-400' : 'text-zinc-500'}`} />
                  <div>
                    <span className="text-xs font-semibold text-white block">
                      Eulerian Video Magnification (EVM)
                    </span>
                    <p className="text-[11px] text-zinc-400">
                      Exaggerates subtle vascular pulse flushes with a dynamic capillary glow overlay
                    </p>
                  </div>
                </div>
                <div
                  className={`w-9 h-5 rounded-full transition-colors relative shrink-0 ${
                    evmEnabled ? 'bg-rose-600' : 'bg-[#202229] border border-[#2e323b]'
                  }`}
                >
                  <div
                    className={`w-4 h-4 rounded-full bg-white transition-transform absolute top-0.5 ${
                      evmEnabled ? 'left-4' : 'left-0.5'
                    }`}
                  />
                </div>
              </button>

              <button
                onClick={onToggleDemoMode}
                className={`w-full p-3 rounded-xl border flex items-center justify-between gap-3 text-left transition-all min-h-[48px] cursor-pointer ${
                  isDemoMode
                    ? 'bg-amber-950/30 border-amber-500/50'
                    : 'bg-[#141519] border-[#262932] hover:border-[#3f4452]'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Sparkles className={`w-4 h-4 ${isDemoMode ? 'text-amber-400' : 'text-zinc-500'}`} />
                  <div>
                    <span className="text-xs font-semibold text-white block">
                      Synthetic Pulse Simulation (Demo)
                    </span>
                    <p className="text-[11px] text-zinc-400">
                      Simulate a 72.0 BPM human capillary pulse signal without camera hardware
                    </p>
                  </div>
                </div>
                <div
                  className={`w-9 h-5 rounded-full transition-colors relative shrink-0 ${
                    isDemoMode ? 'bg-amber-500' : 'bg-[#202229] border border-[#2e323b]'
                  }`}
                >
                  <div
                    className={`w-4 h-4 rounded-full bg-white transition-transform absolute top-0.5 ${
                      isDemoMode ? 'left-4' : 'left-0.5'
                    }`}
                  />
                </div>
              </button>
            </div>
          </div>

          {/* Section 6: Real-Life Environmental Robustness */}
          <div className="p-3.5 rounded-xl bg-[#141519] border border-[#262932] space-y-2">
            <span className="text-xs font-semibold text-white font-mono flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span>Real-Life Artifact Robustness</span>
            </span>
            <div className="space-y-1.5 text-[11px] text-zinc-400 leading-relaxed font-mono">
              <div className="flex items-start gap-1.5">
                <span className="text-emerald-400 font-bold">•</span>
                <span><strong className="text-zinc-200">Subtle Head Movement:</strong> Adaptive landmark filtering locks ROI boundaries against micro-tremors, with physiological inertia tracking.</span>
              </div>
              <div className="flex items-start gap-1.5">
                <span className="text-sky-400 font-bold">•</span>
                <span><strong className="text-zinc-200">Light Changes:</strong> 1.6s sliding sub-window overlap-add continuously detrends ambient lighting drift and exposure adjustments.</span>
              </div>
            </div>
          </div>
        </div>

        {/* Drawer Footer */}
        <div className="pt-4 border-t border-[#262932] mt-6 flex justify-end">
          <button
            onClick={onClose}
            className="w-full sm:w-auto px-6 py-2.5 rounded-xl bg-[#202229] hover:bg-[#282a33] text-white border border-[#2e323b] font-mono text-xs font-semibold transition-all min-h-[44px] cursor-pointer"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
