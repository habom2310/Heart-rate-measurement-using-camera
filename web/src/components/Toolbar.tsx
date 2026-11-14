import React from 'react';
import {
  Download,
  FileSpreadsheet,
  Layers,
  RefreshCw,
  Sliders,
  Sparkles,
} from 'lucide-react';
import type { RPPGAlgorithmType, ROIMode } from '../types/rppg';

interface ToolbarProps {
  algorithm: RPPGAlgorithmType;
  bufferCapacity: number;
  onOpenSettings: () => void;
  showROIs: boolean;
  onToggleROIs: () => void;
  roiMode: ROIMode;
  onCycleROIMode?: () => void;
  isDemoMode: boolean;
  onToggleDemoMode: () => void;
  onRequestReset: () => void;
  onExportCSV: () => void;
  onExportJSON: () => void;
  recordCount: number;
}

export const Toolbar: React.FC<ToolbarProps> = ({
  algorithm,
  bufferCapacity,
  onOpenSettings,
  showROIs,
  onToggleROIs,
  roiMode,
  onCycleROIMode,
  isDemoMode,
  onToggleDemoMode,
  onRequestReset,
  onExportCSV,
  onExportJSON,
  recordCount,
}) => {
  return (
    <div className="bg-[#18191e] border border-[#262932] rounded-xl p-2.5 sm:p-3 shadow-sm flex flex-wrap items-center justify-between gap-2.5">
      {/* Quick DSP & Mode Triggers */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Open Full DSP Settings Drawer */}
        <button
          onClick={onOpenSettings}
          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#202229] border border-[#2e323b] hover:border-[#3f4452] hover:bg-[#282a33] text-xs font-mono transition-all min-h-[44px] sm:min-h-[38px] cursor-pointer"
          title="Open DSP Algorithm and Buffer Settings"
        >
          <Sliders className="w-4 h-4 text-zinc-400 shrink-0" />
          <span className="text-zinc-300 font-semibold">DSP:</span>
          <span className="text-rose-400 font-bold">{algorithm}</span>
          <span className="text-zinc-600">•</span>
          <span className="text-zinc-400">{bufferCapacity}f</span>
        </button>

        {/* Quick ROI Sampling Mode Switcher */}
        <button
          onClick={onCycleROIMode || onToggleROIs}
          className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-mono transition-all border min-h-[44px] sm:min-h-[38px] cursor-pointer ${
            showROIs
              ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
              : 'bg-[#202229] text-zinc-400 border-[#2e323b] hover:text-[#f4f4f6]'
          }`}
          title="Click to cycle ROI Sampling Region: Cheeks -> Forehead -> Full Face"
        >
          <Layers className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <span>ROIs:</span>
          <span className="font-bold text-white">
            {roiMode === 'cheeks' ? 'Cheeks' : roiMode === 'forehead' ? 'Forehead' : 'Full'}
          </span>
        </button>

        {/* Quick Demo Mode Toggle */}
        <button
          onClick={onToggleDemoMode}
          className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-mono transition-all border min-h-[44px] sm:min-h-[38px] cursor-pointer ${
            isDemoMode
              ? 'bg-amber-950/40 border-amber-500/40 text-amber-300'
              : 'bg-[#202229] text-zinc-400 border-[#2e323b] hover:text-[#f4f4f6]'
          }`}
          title="Toggle Synthetic 72 BPM Pulse Simulation"
        >
          <Sparkles className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <span>{isDemoMode ? 'Demo Active' : 'Demo Sim'}</span>
        </button>
      </div>

      {/* Export & Reset Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={onRequestReset}
          className="flex items-center justify-center p-2.5 rounded-lg bg-[#202229] text-zinc-400 border border-[#2e323b] hover:text-rose-400 hover:border-rose-900/60 transition-all min-h-[44px] min-w-[44px] cursor-pointer"
          title="Reset telemetry buffer and clear session"
          aria-label="Reset telemetry buffer"
        >
          <RefreshCw className="w-4 h-4" />
        </button>

        <button
          onClick={onExportCSV}
          disabled={recordCount === 0}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-[#202229] text-emerald-400 border border-emerald-900/50 hover:bg-emerald-950/30 hover:border-emerald-600 disabled:opacity-30 disabled:pointer-events-none transition-all text-xs font-mono min-h-[44px] sm:min-h-[38px] cursor-pointer"
          title="Download timeseries telemetry as CSV"
        >
          <FileSpreadsheet className="w-4 h-4 shrink-0" />
          <span>CSV ({recordCount})</span>
        </button>

        <button
          onClick={onExportJSON}
          disabled={recordCount === 0}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-[#202229] text-sky-400 border border-sky-900/50 hover:bg-sky-950/30 hover:border-sky-600 disabled:opacity-30 disabled:pointer-events-none transition-all text-xs font-mono min-h-[44px] sm:min-h-[38px] cursor-pointer"
          title="Download complete telemetry dataset as JSON"
        >
          <Download className="w-4 h-4 shrink-0" />
          <span>JSON</span>
        </button>
      </div>
    </div>
  );
};
