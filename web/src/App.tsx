import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Heart,
  Info,
  ShieldCheck,
  Sliders,
} from 'lucide-react';
import { CameraFeed } from './components/CameraFeed';
import { DspSettingsDrawer } from './components/DspSettingsDrawer';
import { ResetConfirmModal } from './components/ResetConfirmModal';
import { SpectrumChart } from './components/SpectrumChart';
import { Toolbar } from './components/Toolbar';
import { VitalCards } from './components/VitalCards';
import { WaveformChart } from './components/WaveformChart';
import type {
  HeartRateResult,
  RGBFrame,
  ROIMode,
  RPPGAlgorithmType,
  SessionDataRecord,
  WorkerMessageFromWorker,
  WorkerMessageToWorker,
} from './types/rppg';
import { downloadCSV, downloadJSON } from './utils/export';

export const App: React.FC = () => {
  // DSP & Visualization state
  const [algorithm, setAlgorithm] = useState<RPPGAlgorithmType>('POS');
  const [bufferCapacity, setBufferCapacity] = useState<number>(150);
  const [showMesh, setShowMesh] = useState<boolean>(true);
  const [showROIs, setShowROIs] = useState<boolean>(true);
  const [roiMode, setRoiMode] = useState<ROIMode>('cheeks'); // Cheeks Only by default (Bangs & Beard safe)
  const [showHUD, setShowHUD] = useState<boolean>(true);
  const [evmEnabled, setEvmEnabled] = useState<boolean>(false);
  const [isDemoMode, setIsDemoMode] = useState<boolean>(false);

  // UI Modals & Drawers
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [isResetModalOpen, setIsResetModalOpen] = useState<boolean>(false);

  // Realtime results from Web Worker
  const [latestResult, setLatestResult] = useState<HeartRateResult | null>(null);
  const latestResultRef = useRef<HeartRateResult | null>(null);
  const [bufferFill, setBufferFill] = useState<number>(0);
  const [fps, setFps] = useState<number>(30);

  // Session recording data for export
  const sessionRecordsRef = useRef<SessionDataRecord[]>([]);
  const [recordCount, setRecordCount] = useState<number>(0);

  // DSP Web Worker reference
  const workerRef = useRef<Worker | null>(null);

  // Initialize Web Worker
  useEffect(() => {
    const worker = new Worker(new URL('./workers/rppgWorker.ts', import.meta.url), {
      type: 'module',
    });

    worker.onmessage = (event: MessageEvent<WorkerMessageFromWorker>) => {
      const msg = event.data;
      if (msg.type === 'RESULTS') {
        const res = msg.data;
        latestResultRef.current = res;
        setLatestResult(res);
        setBufferFill(res.bufferFill);
        setFps(res.fps);
      }
    };

    workerRef.current = worker;

    // Send initial configuration
    worker.postMessage({
      type: 'SET_ALGORITHM',
      algorithm: 'POS',
    } as WorkerMessageToWorker);

    worker.postMessage({
      type: 'SET_BUFFER_SIZE',
      size: 150,
    } as WorkerMessageToWorker);

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  // Update Worker on algorithm change
  const handleAlgorithmChange = (newAlg: RPPGAlgorithmType) => {
    setAlgorithm(newAlg);
    workerRef.current?.postMessage({
      type: 'SET_ALGORITHM',
      algorithm: newAlg,
    } as WorkerMessageToWorker);
  };

  // Update Worker on buffer size change
  const handleBufferSizeChange = (newSize: number) => {
    setBufferCapacity(newSize);
    workerRef.current?.postMessage({
      type: 'SET_BUFFER_SIZE',
      size: newSize,
    } as WorkerMessageToWorker);
  };

  // Reset buffers
  const handleReset = () => {
    workerRef.current?.postMessage({ type: 'RESET' } as WorkerMessageToWorker);
    sessionRecordsRef.current = [];
    latestResultRef.current = null;
    setRecordCount(0);
    setLatestResult(null);
    setBufferFill(0);
  };

  // Cycle ROI sampling region (Cheeks -> Forehead + Cheeks -> Forehead Only)
  const handleCycleROIMode = () => {
    setRoiMode((prev) => {
      if (prev === 'cheeks') return 'all';
      if (prev === 'all') return 'forehead';
      return 'cheeks';
    });
  };

  // Frame received from CameraFeed
  const handleFrame = useCallback(
    (frame: RGBFrame) => {
      if (workerRef.current) {
        workerRef.current.postMessage({
          type: 'FRAME',
          data: frame,
        } as WorkerMessageToWorker);
      }

      const res = latestResultRef.current;
      const currentPulse =
        res && res.filteredPulse.length > 0
          ? res.filteredPulse[res.filteredPulse.length - 1]
          : 0;

      sessionRecordsRef.current.push({
        timestamp: frame.timestamp,
        timeSec: frame.timestamp / 1000.0,
        r: frame.r,
        g: frame.g,
        b: frame.b,
        pulse: currentPulse,
        bpm: res?.bpm || 0,
        snr: res?.snr || -30,
        quality: res?.quality || 'Low',
      });

      if (sessionRecordsRef.current.length > 10000) {
        sessionRecordsRef.current.shift();
      }

      // Throttle record count state updates to once per 30 frames (~1 Hz) to prevent 30 FPS React re-render thrashing
      if (sessionRecordsRef.current.length % 30 === 0) {
        setRecordCount(sessionRecordsRef.current.length);
      }
    },
    []
  );

  const handleExportCSV = () => {
    downloadCSV(sessionRecordsRef.current);
  };

  const handleExportJSON = () => {
    downloadJSON(sessionRecordsRef.current, latestResult);
  };

  // Status computation
  const getStatus = () => {
    if (isDemoMode) {
      return { text: 'DEMO SIMULATION', color: 'bg-amber-500/15 text-amber-300 border-amber-500/30' };
    }
    if (bufferFill >= 30) {
      return { text: 'TELEMETRY LOCKED', color: 'bg-rose-500/15 text-rose-300 border-rose-500/30' };
    }
    if (bufferFill > 0) {
      return { text: 'BUFFERING SIGNAL', color: 'bg-sky-500/15 text-sky-300 border-sky-500/30' };
    }
    return { text: 'STANDBY', color: 'bg-[#202229] text-zinc-400 border-[#2e323b]' };
  };

  const status = getStatus();

  return (
    <div className="min-h-screen bg-[#111215] text-[#f4f4f6] flex flex-col font-sans selection:bg-rose-500/30 selection:text-white">
      {/* Top Navigation Header */}
      <header className="border-b border-[#262932] bg-[#18191e]/90 backdrop-blur-md sticky top-0 z-30 px-4 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-rose-500/15 border border-rose-500/30 flex items-center justify-center shrink-0">
              <Heart className="w-4 h-4 text-rose-500 fill-rose-500" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm sm:text-base font-bold tracking-tight text-white font-mono">
                  rPPG PulseSense
                </h1>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#202229] text-zinc-400 border border-[#2e323b] hidden sm:inline">
                  v2.0 WebApp
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 hidden md:block">
                Contactless rPPG Camera Heart-Rate &amp; HRV Telemetry
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-3">
            {/* 100% On-Device Local Privacy Pill */}
            <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-950/40 border border-emerald-500/30 text-[11px] font-mono text-emerald-400">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>100% On-Device</span>
            </div>

            {/* Live System Status Pill */}
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-mono font-medium ${status.color}`}
            >
              <span className="w-2 h-2 rounded-full bg-current animate-pulse" />
              <span>{status.text}</span>
            </div>

            {/* Quick Header DSP Settings Trigger */}
            <button
              onClick={() => setIsSettingsOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#202229] hover:bg-[#282a33] text-[#f4f4f6] text-xs font-mono border border-[#2e323b] transition-all min-h-[38px] cursor-pointer"
              title="Open DSP Algorithm and Filter Settings"
            >
              <Sliders className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
              <span className="hidden sm:inline">DSP Parameters</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Dashboard */}
      <main className="flex-1 max-w-7xl mx-auto w-full p-3 sm:p-4 lg:p-6 space-y-4">
        {/* Compact Quick Action Toolbar */}
        <Toolbar
          algorithm={algorithm}
          bufferCapacity={bufferCapacity}
          onOpenSettings={() => setIsSettingsOpen(true)}
          showROIs={showROIs}
          onToggleROIs={() => setShowROIs(!showROIs)}
          roiMode={roiMode}
          onCycleROIMode={handleCycleROIMode}
          isDemoMode={isDemoMode}
          onToggleDemoMode={() => setIsDemoMode(!isDemoMode)}
          onRequestReset={() => setIsResetModalOpen(true)}
          onExportCSV={handleExportCSV}
          onExportJSON={handleExportJSON}
          recordCount={recordCount}
        />

        {/* Responsive Grid: Left Column (Camera + Vitals), Right Column (Waveforms + Spectrum) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* Left Column (5 cols on desktop): Camera Viewport + Vital Metrics */}
          <div className="lg:col-span-5 flex flex-col space-y-4">
            {/* Camera Viewport (First item on mobile, no scrolling needed!) */}
            <CameraFeed
              onFrame={handleFrame}
              showMesh={showMesh}
              showROIs={showROIs}
              roiMode={roiMode}
              showHUD={showHUD}
              evmEnabled={evmEnabled}
              isDemoMode={isDemoMode}
              onToggleDemoMode={() => setIsDemoMode(!isDemoMode)}
              currentBpm={latestResult?.bpm || 70}
              motionCompensated={latestResult?.motionCompensated}
              lightCompensated={latestResult?.lightCompensated}
            />

            {/* Vital Metrics Summary Cards directly beneath Camera */}
            <VitalCards
              result={latestResult}
              bufferFill={bufferFill}
              bufferCapacity={bufferCapacity}
              fps={fps}
            />
          </div>

          {/* Right Column (7 cols on desktop): Biomedical Oscilloscope & Spectrum */}
          <div className="lg:col-span-7 flex flex-col space-y-4">
            <WaveformChart
              filteredPulse={latestResult?.filteredPulse || []}
              bpm={latestResult?.bpm || 0}
            />
            <SpectrumChart
              frequencies={latestResult?.frequencies || []}
              psd={latestResult?.psd || []}
              peakBpm={latestResult?.bpm || 0}
            />

            {/* Measurement Guidance Card */}
            <div className="bg-[#18191e] border border-[#262932] rounded-xl p-3.5 text-xs text-[#9ca3af] flex items-start gap-2.5">
              <Info className="w-4 h-4 text-zinc-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <span className="font-semibold text-white">Clinical Telemetry Guidance:</span>
                <p className="text-[11px] text-[#9ca3af] leading-relaxed">
                  Sit still under steady, diffuse illumination. Ensure cheekbones (and forehead if enabled) are inside the green ROI hulls. High SNR (&gt;6 dB) indicates robust hemodynamic signal lock.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* DSP Settings Slide-Over Drawer */}
      <DspSettingsDrawer
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        algorithm={algorithm}
        onAlgorithmChange={handleAlgorithmChange}
        roiMode={roiMode}
        onRoiModeChange={setRoiMode}
        bufferCapacity={bufferCapacity}
        onBufferSizeChange={handleBufferSizeChange}
        showMesh={showMesh}
        onToggleMesh={() => setShowMesh(!showMesh)}
        showROIs={showROIs}
        onToggleROIs={() => setShowROIs(!showROIs)}
        showHUD={showHUD}
        onToggleHUD={() => setShowHUD(!showHUD)}
        evmEnabled={evmEnabled}
        onToggleEVM={() => setEvmEnabled(!evmEnabled)}
        isDemoMode={isDemoMode}
        onToggleDemoMode={() => setIsDemoMode(!isDemoMode)}
      />

      {/* Reset Confirmation Dialog Modal */}
      <ResetConfirmModal
        isOpen={isResetModalOpen}
        onClose={() => setIsResetModalOpen(false)}
        onConfirm={handleReset}
        recordCount={recordCount}
      />

      {/* Footer */}
      <footer className="border-t border-[#262932] bg-[#111215] py-3 px-4 text-center text-xs text-zinc-500 font-mono">
        rPPG WebApp v2.0 • WebAssembly &amp; Web Workers • 100% On-Device Telemetry
      </footer>
    </div>
  );
};

export default App;
