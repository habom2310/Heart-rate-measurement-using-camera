import React from 'react';
import { AlertCircle, RefreshCw, X } from 'lucide-react';

interface ResetConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  recordCount: number;
}

export const ResetConfirmModal: React.FC<ResetConfirmModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  recordCount,
}) => {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#111215]/80 backdrop-blur-sm"
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="reset-modal-title"
      aria-describedby="reset-modal-desc"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md bg-[#18191e] border border-[#262932] rounded-2xl p-6 shadow-2xl space-y-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between">
          <div className="w-12 h-12 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400">
            <AlertCircle className="w-6 h-6" />
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-[#202229] transition-all min-h-[44px] min-w-[44px] flex items-center justify-center cursor-pointer"
            aria-label="Close dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div>
          <h3 id="reset-modal-title" className="text-base font-bold text-white font-mono">
            Reset Telemetry Session?
          </h3>
          <p id="reset-modal-desc" className="text-xs text-zinc-400 mt-2 leading-relaxed">
            This will clear{' '}
            <span className="text-white font-mono font-bold">{recordCount}</span> recorded
            hemodynamic pulse samples and re-initialize real-time signal buffers. Live camera streaming will continue uninterrupted.
          </p>
        </div>

        <div className="flex items-center justify-end gap-3 pt-2">
          <button
            onClick={onClose}
            className="px-4 py-2.5 rounded-xl bg-[#202229] hover:bg-[#282a33] text-zinc-300 border border-[#2e323b] text-xs font-mono font-medium transition-all min-h-[44px] cursor-pointer"
          >
            Keep Telemetry
          </button>
          <button
            onClick={() => {
              onConfirm();
              onClose();
            }}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-mono font-bold shadow-md shadow-rose-950/50 transition-all min-h-[44px] cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Confirm Reset</span>
          </button>
        </div>
      </div>
    </div>
  );
};
