import type { HeartRateResult, SessionDataRecord } from '../types/rppg';

export function downloadCSV(records: SessionDataRecord[], filename = 'rppg_session_data.csv'): void {
  if (records.length === 0) return;

  const headers = ['Timestamp_ms', 'Time_s', 'R_mean', 'G_mean', 'B_mean', 'Filtered_Pulse', 'BPM', 'SNR_dB', 'Quality'];
  const rows = records.map((r) => [
    r.timestamp,
    r.timeSec.toFixed(3),
    r.r.toFixed(2),
    r.g.toFixed(2),
    r.b.toFixed(2),
    r.pulse.toFixed(5),
    r.bpm.toFixed(1),
    r.snr.toFixed(2),
    r.quality,
  ]);

  const csvContent = [headers.join(','), ...rows.map((row) => row.join(','))].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function downloadJSON(
  records: SessionDataRecord[],
  latestResult: HeartRateResult | null,
  filename = 'rppg_session_report.json',
): void {
  const exportPayload = {
    exportedAt: new Date().toISOString(),
    totalSamples: records.length,
    finalResult: latestResult,
    timeseries: records,
  };

  const jsonContent = JSON.stringify(exportPayload, null, 2);
  const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
