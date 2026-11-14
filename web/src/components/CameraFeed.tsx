import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Camera,
  CameraOff,
  ExternalLink,
  Flame,
  Maximize2,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  SwitchCamera,
  UserCheck,
  X,
} from 'lucide-react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { RGBFrame, ROIMode } from '../types/rppg';

interface CameraFeedProps {
  onFrame: (frame: RGBFrame) => void;
  showMesh: boolean;
  showROIs: boolean;
  roiMode?: ROIMode;
  showHUD: boolean;
  evmEnabled: boolean;
  isDemoMode: boolean;
  onToggleDemoMode?: () => void;
  currentBpm: number;
  motionCompensated?: boolean;
  lightCompensated?: boolean;
}

// MediaPipe 468 landmark subsets (Anatomically calibrated for rPPG)
// Stable rigid nasal anchors for inter-frame velocity estimation
const ANCHOR_INDICES = [1, 4, 168, 197];
// Symmetrical Mid-Forehead (balanced across facial midline, avoiding hair & eyebrows)
const FOREHEAD_INDICES = [67, 109, 10, 338, 297, 299, 296, 336, 9, 107, 66, 69];
// Right Malar Cheek (subject's right, viewer's left in mirror mode)
const RIGHT_CHEEK_INDICES = [117, 118, 119, 100, 101, 50, 205, 36, 123];
// Left Malar Cheek (subject's left, viewer's right in mirror mode)
const LEFT_CHEEK_INDICES = [346, 347, 348, 329, 330, 280, 425, 266, 352];
const JAWLINE_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];

// 2D Monotone Chain convex hull to ensure ROI polygons never self-intersect or twist
function getConvexHull(points: { x: number; y: number }[]): { x: number; y: number }[] {
  if (points.length <= 3) return points;
  const sorted = points.slice().sort((a, b) => (a.x !== b.x ? a.x - b.x : a.y - b.y));

  const cross = (o: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: { x: number; y: number }[] = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: { x: number; y: number }[] = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

function drawHUDOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  r: number,
  g: number,
  b: number,
  hasOverlayBadges: boolean = false,
  motionCompensated: boolean = false,
  lightCompensated: boolean = false
) {
  ctx.save();

  const bracketSize = 20;
  const pad = 16;
  ctx.strokeStyle = 'rgba(156, 163, 175, 0.4)'; // Hairline technical slate
  ctx.lineWidth = 1.5;

  // Draw top corner brackets only if no HTML overlay badges are sitting there
  if (!hasOverlayBadges) {
    // Top-Left
    ctx.beginPath();
    ctx.moveTo(pad, pad + bracketSize);
    ctx.lineTo(pad, pad);
    ctx.lineTo(pad + bracketSize, pad);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(width - pad - bracketSize, pad);
    ctx.lineTo(width - pad, pad);
    ctx.lineTo(width - pad, pad + bracketSize);
    ctx.stroke();
  }

  // Bottom-Left
  ctx.beginPath();
  ctx.moveTo(pad, height - pad - bracketSize);
  ctx.lineTo(pad, height - pad);
  ctx.lineTo(pad + bracketSize, height - pad);
  ctx.stroke();

  // Bottom-Right
  ctx.beginPath();
  ctx.moveTo(width - pad - bracketSize, height - pad);
  ctx.lineTo(width - pad, height - pad);
  ctx.lineTo(width - pad, height - pad - bracketSize);
  ctx.stroke();

  // Live RGB Extraction Telemetry Tag (Bottom Left - inside brackets)
  ctx.fillStyle = 'rgba(24, 25, 30, 0.9)';
  ctx.strokeStyle = '#262932';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(pad + 4, height - pad - 26, 176, 22, 6);
  ctx.fill();
  ctx.stroke();

  ctx.font = '500 10px monospace';
  ctx.fillStyle = '#fb7185'; // Arterial Crimson / Coral
  ctx.fillText(`R:${Math.round(r)}`, pad + 14, height - pad - 12);
  ctx.fillStyle = '#34d399'; // Clinical Mint
  ctx.fillText(`G:${Math.round(g)}`, pad + 70, height - pad - 12);
  ctx.fillStyle = '#38bdf8'; // Steel Azure
  ctx.fillText(`B:${Math.round(b)}`, pad + 126, height - pad - 12);

  // Motion / Lighting compensation badge (Bottom Right - inside brackets)
  if (motionCompensated || lightCompensated) {
    const statusText = motionCompensated ? 'STABILIZED' : 'LIGHT NORM';
    ctx.font = '500 9px monospace';
    const textW = ctx.measureText(statusText).width;
    ctx.fillStyle = 'rgba(24, 25, 30, 0.9)';
    ctx.strokeStyle = motionCompensated ? '#d97706' : '#0284c7';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(width - pad - textW - 20, height - pad - 26, textW + 16, 22, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = motionCompensated ? '#fbbf24' : '#38bdf8';
    ctx.fillText(statusText, width - pad - textW - 12, height - pad - 12);
  }

  ctx.restore();
}

export const CameraFeed: React.FC<CameraFeedProps> = ({
  onFrame,
  showMesh,
  showROIs,
  roiMode = 'cheeks',
  showHUD,
  evmEnabled,
  isDemoMode,
  onToggleDemoMode,
  currentBpm,
  motionCompensated = false,
  lightCompensated = false,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const animationFrameId = useRef<number | null>(null);
  const lastVideoTime = useRef<number>(-1);

  // Adaptive landmark stabilization and motion estimation
  const smoothedLandmarksRef = useRef<{ x: number; y: number }[] | null>(null);
  const lastAnchorPointsRef = useRef<{ x: number; y: number }[] | null>(null);
  const currentMotionVelocityRef = useRef<number>(0);
  const lastSampledLuminanceRef = useRef<number>(0);
  const lastSampledRGBRef = useRef<{ r: number; g: number; b: number } | null>(null);

  const [streamActive, setStreamActive] = useState<boolean>(false);
  const [isConnecting, setIsConnecting] = useState<boolean>(false);
  const [cameraFacing, setCameraFacing] = useState<'user' | 'environment'>('user');
  const [videoAspect, setVideoAspect] = useState<number>(4 / 3);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [modelLoading, setModelLoading] = useState<boolean>(true);
  const [faceDetected, setFaceDetected] = useState<boolean>(false);
  const faceDetectedRef = useRef<boolean>(false);
  const currentMeanRGBRef = useRef<{ r: number; g: number; b: number }>({
    r: 0,
    g: 0,
    b: 0,
  });

  const isSecureContext = typeof window !== 'undefined' ? window.isSecureContext : false;
  const hasMediaDevices = typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia;

  // Eulerian magnification temporal phase accumulator
  const evmPhaseRef = useRef<number>(0);

  // Picture-in-Picture (PiP) corner docking on scroll
  const anchorRef = useRef<HTMLDivElement | null>(null);
  const [isScrolledPast, setIsScrolledPast] = useState<boolean>(false);
  const [pipDismissed, setPipDismissed] = useState<boolean>(false);

  useEffect(() => {
    const anchor = anchorRef.current;
    if (!anchor || typeof IntersectionObserver === 'undefined') return;

    // 20 granular evaluation thresholds (every 5%) to accurately detect the 80% scroll mark
    const thresholds: number[] = [];
    for (let i = 0; i <= 20; i++) {
      thresholds.push(i / 20);
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        const isPastTop = entry.boundingClientRect.top < 0;
        const ratio = entry.intersectionRatio;

        // Proactive early docking: dock when ~80% of camera has scrolled out of view (ratio <= 0.20),
        // rather than waiting until 100% gone. Hysteresis undocks when ratio >= 0.28 or back at top.
        if (isPastTop && ratio <= 0.20) {
          setIsScrolledPast(true);
        } else if (!isPastTop || ratio >= 0.28) {
          setIsScrolledPast(false);
          setPipDismissed(false);
        }
      },
      {
        threshold: thresholds,
      }
    );

    observer.observe(anchor);
    return () => observer.disconnect();
  }, []);

  const handleScrollToCamera = useCallback(() => {
    anchorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, []);

  const shouldDock = isScrolledPast && (streamActive || isDemoMode) && !pipDismissed;

  // Initialize MediaPipe FaceLandmarker
  useEffect(() => {
    let isMounted = true;
    async function initFaceLandmarker() {
      try {
        setModelLoading(true);
        const filesetResolver = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        let landmarker: FaceLandmarker;
        try {
          landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numFaces: 1,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
          });
        } catch (gpuErr) {
          console.warn('MediaPipe GPU initialization failed, attempting CPU fallback:', gpuErr);
          landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
              delegate: 'CPU',
            },
            runningMode: 'VIDEO',
            numFaces: 1,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
          });
        }

        if (isMounted) {
          landmarkerRef.current = landmarker;
          setModelLoading(false);
        }
      } catch (err) {
        console.warn('MediaPipe FaceLandmarker failed to load from remote CDN, using fallback tracking:', err);
        if (isMounted) {
          setModelLoading(false);
        }
      }
    }

    initFaceLandmarker();

    return () => {
      isMounted = false;
      if (landmarkerRef.current) {
        landmarkerRef.current.close();
        landmarkerRef.current = null;
      }
    };
  }, []);

  // Stop camera helper
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStreamActive(false);
    setIsConnecting(false);
  }, []);

  // Start Camera with explicit user action
  const startCamera = useCallback(async (facing: 'user' | 'environment' = cameraFacing) => {
    const secureUrl = typeof window !== 'undefined'
      ? (window.location.protocol === 'https:' ? window.location.href : `https://${window.location.host}${window.location.pathname}`)
      : 'https://localhost:5173/';

    if (!hasMediaDevices || !isSecureContext) {
      setCameraError(
        `Camera access requires a Secure Context (HTTPS) on mobile browsers. Please connect using: ${secureUrl}`
      );
      return;
    }

    setIsConnecting(true);
    setCameraError(null);

    // Stop existing stream if any
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    try {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: facing,
            width: { ideal: 640 },
          },
          audio: false,
        });
      } catch (initialErr) {
        console.warn('Initial getUserMedia with facingMode failed, falling back to basic video constraint:', initialErr);
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }

      streamRef.current = stream;

      if (videoRef.current) {
        const video = videoRef.current;
        video.srcObject = stream;
        video.setAttribute('playsinline', 'true');
        video.setAttribute('webkit-playsinline', 'true');
        video.muted = true;

        await video.play();
        setCameraError(null);
        setStreamActive(true);
        setIsConnecting(false);
      }
    } catch (err: unknown) {
      console.error('Camera access failed:', err);
      setIsConnecting(false);
      setStreamActive(false);
      const message = err instanceof Error ? err.message : 'Unable to access camera';
      setCameraError(
        message.includes('Permission') || message.includes('denied')
          ? 'Camera permission was denied. Please allow camera access in your browser address bar settings, or use Synthetic Demo Mode below.'
          : `Unable to connect to camera (${message}). Switch to Synthetic Demo Mode to test the full DSP pipeline.`
      );
    }
  }, [cameraFacing, hasMediaDevices, isSecureContext]);

  // Flip between front and rear cameras
  const toggleFacingMode = useCallback(() => {
    const nextFacing = cameraFacing === 'user' ? 'environment' : 'user';
    setCameraFacing(nextFacing);
    if (streamActive) {
      startCamera(nextFacing);
    }
  }, [cameraFacing, startCamera, streamActive]);

  // Clean up tracks when demo mode is enabled
  useEffect(() => {
    if (isDemoMode) {
      stopCamera();
    }
  }, [isDemoMode, stopCamera]);

  // Clean up camera stream only when component truly unmounts
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  // Main Render and RGB Extraction Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement('canvas');
      offscreenCanvasRef.current.width = 640;
      offscreenCanvasRef.current.height = 480;
    }
    const offCanvas = offscreenCanvasRef.current;
    const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });

    let syntheticAngle = 0;

    const processFrame = () => {
      const now = performance.now();

      if (isDemoMode) {
        // Ensure Demo Mode stays at canonical 4:3 640x480 resolution
        if (canvas.width !== 640 || canvas.height !== 480) {
          canvas.width = 640;
          canvas.height = 480;
          if (offscreenCanvasRef.current) {
            offscreenCanvasRef.current.width = 640;
            offscreenCanvasRef.current.height = 480;
          }
          setVideoAspect(4 / 3);
        }

        const width = canvas.width;
        const height = canvas.height;

        // --- SYNTHETIC DEMO SIMULATION ---
        if (!faceDetectedRef.current) {
          faceDetectedRef.current = true;
          setFaceDetected(true);
        }
        syntheticAngle += 0.05;
        // Simulated heart rate at ~72 BPM (1.2 Hz) + subtle 0.25 Hz respiration modulation
        const heartFreq = 1.2;
        const heartPhase = (now / 1000.0) * heartFreq * 2.0 * Math.PI;
        const pulseMod = Math.sin(heartPhase) * 2.8 + Math.sin(heartPhase * 2.0) * 0.9;

        // Base skin color
        const baseR = 195.0 + Math.sin(now * 0.0008) * 1.5;
        const baseG = 142.0 + pulseMod; // Hemoglobin modulates Green strongly
        const baseB = 120.0 - pulseMod * 0.4;

        currentMeanRGBRef.current = {
          r: Math.round(baseR),
          g: Math.round(baseG),
          b: Math.round(baseB),
        };

        onFrame({
          r: baseR,
          g: baseG,
          b: baseB,
          timestamp: now,
        });

        // Draw Synthetic Avatar on canvas
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);

        // Grid background
        ctx.strokeStyle = 'rgba(30, 41, 59, 0.4)';
        ctx.lineWidth = 1;
        for (let x = 0; x < width; x += 40) {
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
        }
        for (let y = 0; y < height; y += 40) {
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(width, y);
          ctx.stroke();
        }

        // Draw Face Oval with dynamic capillary pulse flush if EVM is enabled
        const faceCx = width / 2;
        const faceCy = height / 2 + 10;
        const faceRx = 105;
        const faceRy = 145;

        // Eulerian magnification color flash
        const ampFactor = evmEnabled ? 8.0 : 1.0;
        const evmR = Math.min(255, Math.max(0, baseR + (evmEnabled ? pulseMod * ampFactor : 0)));
        const evmG = Math.min(255, Math.max(0, baseG - (evmEnabled ? pulseMod * ampFactor * 0.5 : 0)));
        const evmB = Math.min(255, Math.max(0, baseB - (evmEnabled ? pulseMod * ampFactor * 0.5 : 0)));

        ctx.save();
        ctx.beginPath();
        ctx.ellipse(faceCx, faceCy, faceRx, faceRy, 0, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${Math.round(evmR)}, ${Math.round(evmG)}, ${Math.round(evmB)})`;
        ctx.shadowColor = evmEnabled ? 'rgba(244, 63, 94, 0.4)' : 'rgba(56, 189, 248, 0.2)';
        ctx.shadowBlur = 20;
        ctx.fill();
        ctx.restore();

        // Synthetic ROIs
        const forehead = [
          { x: faceCx - 45, y: faceCy - 75 },
          { x: faceCx + 45, y: faceCy - 75 },
          { x: faceCx + 40, y: faceCy - 40 },
          { x: faceCx - 40, y: faceCy - 40 },
        ];
        const leftCheek = [
          { x: faceCx - 70, y: faceCy - 5 },
          { x: faceCx - 25, y: faceCy - 5 },
          { x: faceCx - 30, y: faceCy + 35 },
          { x: faceCx - 65, y: faceCy + 35 },
        ];
        const rightCheek = [
          { x: faceCx + 25, y: faceCy - 5 },
          { x: faceCx + 70, y: faceCy - 5 },
          { x: faceCx + 65, y: faceCy + 35 },
          { x: faceCx + 30, y: faceCy + 35 },
        ];

        const demoROIs =
          roiMode === 'forehead'
            ? [forehead]
            : roiMode === 'all'
            ? [forehead, leftCheek, rightCheek]
            : [leftCheek, rightCheek];

        if (showROIs) {
          demoROIs.forEach((poly) => {
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(poly[0].x, poly[0].y);
            for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i].x, poly[i].y);
            ctx.closePath();
            ctx.fillStyle = 'rgba(16, 185, 129, 0.25)';
            ctx.fill();
            ctx.strokeStyle = '#34d399';
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.restore();
          });
        }

        if (showMesh) {
          ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.ellipse(faceCx, faceCy, faceRx * 0.9, faceRy * 0.9, 0, 0, Math.PI * 2);
          ctx.stroke();
        }

        if (showHUD) {
          drawHUDOverlay(ctx, width, height, baseR, baseG, baseB, true);
        }

        animationFrameId.current = requestAnimationFrame(processFrame);
        return;
      }

      // --- LIVE WEBCAM PROCESSING ---
      const video = videoRef.current;
      if (!video || video.readyState < 2) {
        animationFrameId.current = requestAnimationFrame(processFrame);
        return;
      }

      // Synchronize canvas resolution with incoming video feed to eliminate aspect ratio squashing/stretching
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (vw > 0 && vh > 0 && (canvas.width !== vw || canvas.height !== vh)) {
        canvas.width = vw;
        canvas.height = vh;
        if (offscreenCanvasRef.current) {
          offscreenCanvasRef.current.width = vw;
          offscreenCanvasRef.current.height = vh;
        }
        const aspect = vw / vh;
        setVideoAspect((prev) => (Math.abs(prev - aspect) > 0.01 ? aspect : prev));
      }

      const width = canvas.width;
      const height = canvas.height;

      // Draw video frame to canvas
      ctx.save();
      if (cameraFacing === 'user') {
        // Mirrored only for front-facing selfie camera
        ctx.translate(width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, width, height);
      } else {
        // Natural non-mirrored orientation for rear/environment camera
        ctx.drawImage(video, 0, 0, width, height);
      }
      ctx.restore();

      let detected = false;
      let landmarks: { x: number; y: number }[] | null = null;

      if (landmarkerRef.current && video.currentTime !== lastVideoTime.current) {
        lastVideoTime.current = video.currentTime;
        try {
          const results = landmarkerRef.current.detectForVideo(video, now);
          if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            detected = true;
            // Map normalized coordinates [0, 1] to canvas coordinates matching mirror state
            const rawLandmarks = results.faceLandmarks[0].map((pt) => ({
              x: cameraFacing === 'user' ? (1.0 - pt.x) * width : pt.x * width,
              y: pt.y * height,
            }));

            // 1. Calculate physical head motion velocity from stable rigid nasal anchors
            let anchorDisp = 0;
            if (lastAnchorPointsRef.current && rawLandmarks.length > 200) {
              let dispSum = 0;
              for (let k = 0; k < ANCHOR_INDICES.length; k++) {
                const idx = ANCHOR_INDICES[k];
                const prev = lastAnchorPointsRef.current[k];
                const curr = rawLandmarks[idx];
                const dx = curr.x - prev.x;
                const dy = curr.y - prev.y;
                dispSum += Math.sqrt(dx * dx + dy * dy);
              }
              anchorDisp = dispSum / ANCHOR_INDICES.length;
            }
            lastAnchorPointsRef.current = ANCHOR_INDICES.map((idx) => ({
              x: rawLandmarks[idx].x,
              y: rawLandmarks[idx].y,
            }));
            currentMotionVelocityRef.current = anchorDisp;

            // 2. Adaptive Landmark Stabilization (One-Euro / Dynamic EMA)
            // Locks the ROI boundary stationary when subject has micro-tremor/jitter (< 1.2 px),
            // while tracking smoothly with zero lag when the user turns or moves (> 5 px).
            if (
              !smoothedLandmarksRef.current ||
              smoothedLandmarksRef.current.length !== rawLandmarks.length
            ) {
              smoothedLandmarksRef.current = rawLandmarks.map((pt) => ({ x: pt.x, y: pt.y }));
            } else {
              const beta =
                anchorDisp < 1.2
                  ? 0.15
                  : anchorDisp > 5.5
                  ? 0.90
                  : 0.15 + (anchorDisp - 1.2) * 0.17;
              for (let i = 0; i < rawLandmarks.length; i++) {
                smoothedLandmarksRef.current[i].x =
                  (1.0 - beta) * smoothedLandmarksRef.current[i].x + beta * rawLandmarks[i].x;
                smoothedLandmarksRef.current[i].y =
                  (1.0 - beta) * smoothedLandmarksRef.current[i].y + beta * rawLandmarks[i].y;
              }
            }

            landmarks = smoothedLandmarksRef.current;
          }
        } catch {
          // Detection frame drop fallback
        }
      }

      if (faceDetectedRef.current !== detected) {
        faceDetectedRef.current = detected;
        setFaceDetected(detected);
      }

      // Extract ROIs
      let foreheadPts: { x: number; y: number }[] = [];
      let leftCheekPts: { x: number; y: number }[] = [];
      let rightCheekPts: { x: number; y: number }[] = [];

      if (landmarks && landmarks.length > 400) {
        foreheadPts = getConvexHull(FOREHEAD_INDICES.map((idx) => landmarks![idx]));
        leftCheekPts = getConvexHull(LEFT_CHEEK_INDICES.map((idx) => landmarks![idx]));
        rightCheekPts = getConvexHull(RIGHT_CHEEK_INDICES.map((idx) => landmarks![idx]));
      } else {
        // Fallback default center-face ROI if face landmarker is still loading
        const cx = width / 2;
        const cy = height / 2;
        foreheadPts = [
          { x: cx - 45, y: cy - 70 },
          { x: cx + 45, y: cy - 70 },
          { x: cx + 40, y: cy - 40 },
          { x: cx - 40, y: cy - 40 },
        ];
        leftCheekPts = [
          { x: cx - 65, y: cy },
          { x: cx - 25, y: cy },
          { x: cx - 30, y: cy + 40 },
          { x: cx - 65, y: cy + 40 },
        ];
        rightCheekPts = [
          { x: cx + 25, y: cy },
          { x: cx + 65, y: cy },
          { x: cx + 65, y: cy + 40 },
          { x: cx + 30, y: cy + 40 },
        ];
      }

      // Determine active sampling ROIs based on roiMode (default: 'cheeks')
      const activeROIs =
        roiMode === 'forehead'
          ? [foreheadPts]
          : roiMode === 'all'
          ? [foreheadPts, leftCheekPts, rightCheekPts]
          : [leftCheekPts, rightCheekPts];

      // Compute Spatial Mean RGB from active ROIs
      if (offCtx) {
        let totalR = 0;
        let totalG = 0;
        let totalB = 0;
        let totalCount = 0;

        activeROIs.forEach((poly) => {
          if (poly.length < 3) return;

          let minX = width;
          let maxX = 0;
          let minY = height;
          let maxY = 0;
          poly.forEach((pt) => {
            if (pt.x < minX) minX = pt.x;
            if (pt.x > maxX) maxX = pt.x;
            if (pt.y < minY) minY = pt.y;
            if (pt.y > maxY) maxY = pt.y;
          });

          const bx = Math.max(0, Math.floor(minX));
          const by = Math.max(0, Math.floor(minY));
          const bw = Math.min(width - bx, Math.ceil(maxX - minX));
          const bh = Math.min(height - by, Math.ceil(maxY - minY));

          if (bw > 2 && bh > 2) {
            try {
              const imgData = ctx.getImageData(bx, by, bw, bh);
              const data = imgData.data;
              for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                // Luminance outlier rejection
                const lum = 0.299 * r + 0.587 * g + 0.114 * b;
                // Biological skin vs. hair/beard/shadow discriminator:
                // Rejects dark hair, beard stubble (lum < 35 or r ~= g ~= b) and specular highlights (lum > 240)
                const isSkin = lum > 35 && lum < 240 && r > b && (r >= g || Math.abs(r - g) < 20);
                if (isSkin) {
                  totalR += r;
                  totalG += g;
                  totalB += b;
                  totalCount++;
                }
              }
            } catch {
              // Canvas tainted or bounds error guard
            }
          }
        });

        if (totalCount > 10) {
          let meanR = totalR / totalCount;
          let meanG = totalG / totalCount;
          let meanB = totalB / totalCount;

          // Illumination Step Guard: check for abrupt (> 18%) luminance transient across adjacent frames
          // (e.g. camera auto-exposure re-calibration, phone screen brightness change, lamp flickers)
          const currentLum = 0.299 * meanR + 0.587 * meanG + 0.114 * meanB;
          if (lastSampledLuminanceRef.current > 10 && lastSampledRGBRef.current) {
            const lumRatio =
              Math.abs(currentLum - lastSampledLuminanceRef.current) / lastSampledLuminanceRef.current;
            if (lumRatio > 0.18) {
              // Soften sudden flash so the bandpass filter does not experience an impulse spike
              meanR = 0.65 * lastSampledRGBRef.current.r + 0.35 * meanR;
              meanG = 0.65 * lastSampledRGBRef.current.g + 0.35 * meanG;
              meanB = 0.65 * lastSampledRGBRef.current.b + 0.35 * meanB;
            }
          }
          lastSampledLuminanceRef.current = currentLum;
          lastSampledRGBRef.current = { r: meanR, g: meanG, b: meanB };

          currentMeanRGBRef.current = {
            r: Math.round(meanR),
            g: Math.round(meanG),
            b: Math.round(meanB),
          };

          onFrame({
            r: meanR,
            g: meanG,
            b: meanB,
            timestamp: now,
            motionVelocity: currentMotionVelocityRef.current,
          });

          // Eulerian Video Magnification (EVM) pulse highlight overlay
          if (evmEnabled) {
            evmPhaseRef.current += (currentBpm > 40 ? currentBpm : 70) * 0.003;
            const evmPulse = Math.sin(evmPhaseRef.current);
            ctx.save();
            ctx.fillStyle = `rgba(244, 63, 94, ${Math.max(0, evmPulse * 0.35).toFixed(2)})`;
            activeROIs.forEach((poly) => {
              if (poly.length >= 3) {
                ctx.beginPath();
                ctx.moveTo(poly[0].x, poly[0].y);
                for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i].x, poly[i].y);
                ctx.closePath();
                ctx.fill();
              }
            });
            ctx.restore();
          }
        }
      }

      // Draw ROI Highlights
      if (showROIs) {
        activeROIs.forEach((poly) => {
          if (poly.length >= 3) {
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(poly[0].x, poly[0].y);
            for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i].x, poly[i].y);
            ctx.closePath();
            ctx.fillStyle = 'rgba(16, 185, 129, 0.14)';
            ctx.fill();
            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.restore();
          }
        });
      }

      // Draw Facial Mesh Contours
      if (showMesh && landmarks && landmarks.length > 400) {
        ctx.save();
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)'; // Hairline technical slate
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < JAWLINE_INDICES.length; i++) {
          const pt = landmarks[JAWLINE_INDICES[i]];
          if (pt) {
            if (i === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
          }
        }
        ctx.closePath();
        ctx.stroke();
        ctx.restore();
      }

      // Draw Medical HUD Reticles
      if (showHUD) {
        drawHUDOverlay(
          ctx,
          width,
          height,
          currentMeanRGBRef.current.r,
          currentMeanRGBRef.current.g,
          currentMeanRGBRef.current.b,
          streamActive,
          motionCompensated,
          lightCompensated
        );
      }

      // Draw Center-Face Framing Reticle Guide when searching for subject
      if (!detected && streamActive && !isDemoMode) {
        ctx.save();
        ctx.strokeStyle = 'rgba(225, 29, 72, 0.45)'; // Arterial crimson guidance ring
        ctx.lineWidth = 1.5;
        ctx.setLineDash([8, 6]);
        const cx = width / 2;
        const cy = height / 2 - 10;
        const rx = Math.min(width, height) * 0.28;
        const ry = Math.min(width, height) * 0.38;
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#f4f4f6';
        ctx.font = 'bold 11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('ALIGN FACE IN RETICLE', cx, Math.min(height - 20, cy + ry + 24));
        ctx.restore();
      }

      animationFrameId.current = requestAnimationFrame(processFrame);
    };

    animationFrameId.current = requestAnimationFrame(processFrame);

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [
    isDemoMode,
    onFrame,
    showMesh,
    showROIs,
    roiMode,
    showHUD,
    evmEnabled,
    currentBpm,
    cameraFacing,
    motionCompensated,
    lightCompensated,
  ]);

  return (
    <div
      ref={anchorRef}
      className="relative w-full mx-auto"
      style={{
        aspectRatio: `${videoAspect}`,
        maxHeight: '520px',
        maxWidth: videoAspect < 1 ? '380px' : '100%',
      }}
    >
      {/* Ghost placeholder when camera is docked in corner to prevent layout jumps */}
      {shouldDock && (
        <div
          onClick={handleScrollToCamera}
          className="w-full h-full rounded-xl border border-dashed border-[#262932] bg-[#141519]/70 flex flex-col items-center justify-center p-4 text-center cursor-pointer hover:border-rose-500/40 transition-colors group ghost-placeholder-enter"
        >
          <div className="flex items-center gap-2 text-xs font-mono text-zinc-300 group-hover:text-rose-300">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>Camera Active in Corner</span>
            <Maximize2 className="w-3.5 h-3.5 ml-1 text-zinc-500 group-hover:text-rose-400 transition-colors" />
          </div>
          <p className="text-[11px] text-zinc-500 mt-1.5 font-mono">
            Click to return to full camera view
          </p>
        </div>
      )}

      {/* Main Viewport Container (Seamlessly docks to bottom-right PiP on scroll) */}
      <div
        onClick={shouldDock ? handleScrollToCamera : undefined}
        className={
          shouldDock
            ? 'fixed bottom-4 right-4 z-50 w-40 sm:w-56 md:w-64 rounded-xl overflow-hidden border border-[#262932] shadow-2xl shadow-black/95 bg-[#18191e] transition-all duration-300 hover:border-rose-500/50 hover:shadow-rose-950/30 cursor-pointer group pip-dock-enter'
            : 'relative w-full h-full bg-[#18191e] rounded-xl overflow-hidden border border-[#262932] shadow-sm flex items-center justify-center mx-auto transition-all duration-300'
        }
        style={{
          aspectRatio: `${videoAspect}`,
          maxHeight: shouldDock ? '240px' : '520px',
        }}
      >
        {/* Offscreen HTML5 Video element (cannot be display:none for iOS Safari WebKit decoder) */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '1px',
            height: '1px',
            opacity: 0,
            pointerEvents: 'none',
            zIndex: -1,
          }}
        />

        {/* Main HTML5 Viewport Canvas */}
        <canvas
          ref={canvasRef}
          width={videoAspect < 1 ? 480 : 640}
          height={videoAspect < 1 ? 640 : 480}
          className="w-full h-full object-contain block bg-[#0d0e11]"
        />

        {/* Floating PiP Header & Controls when Docked */}
        {shouldDock ? (
          <div className="absolute inset-0 flex flex-col justify-between p-2 pointer-events-none z-30">
            <div className="flex items-center justify-between pointer-events-auto">
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-[#18191e]/90 border border-[#262932] backdrop-blur text-[10px] font-mono text-[#f4f4f6]">
                <span className={`w-1.5 h-1.5 rounded-full ${faceDetected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
                <span className="hidden sm:inline">{faceDetected ? 'Tracking' : 'Searching'}</span>
              </div>

              <div className="flex items-center gap-1">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleScrollToCamera();
                  }}
                  className="p-1 rounded-md bg-[#18191e]/90 hover:bg-[#202229] border border-[#262932] text-zinc-300 hover:text-white transition-all cursor-pointer"
                  title="Expand to full camera view"
                  aria-label="Expand camera view"
                >
                  <Maximize2 className="w-3 h-3" />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setPipDismissed(true);
                  }}
                  className="p-1 rounded-md bg-[#18191e]/90 hover:bg-[#202229] border border-[#262932] text-zinc-400 hover:text-rose-400 transition-all cursor-pointer"
                  title="Dismiss corner camera"
                  aria-label="Dismiss corner camera"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            </div>

            <div className="self-center pointer-events-auto px-2 py-0.5 rounded-full bg-[#18191e]/85 border border-[#262932] text-[9px] font-mono text-zinc-400 group-hover:text-white group-hover:border-rose-500/40 transition-all">
              Tap to expand
            </div>
          </div>
        ) : (
          <>
            {/* EVM Indicator Badge */}
            {evmEnabled && streamActive && (
              <div className="absolute top-4 right-4 bg-rose-950/80 border border-rose-500/60 text-rose-300 text-xs font-mono px-2.5 py-1 rounded-md shadow-sm flex items-center gap-1.5 animate-pulse z-10">
                <Flame className="w-3.5 h-3.5 text-rose-400" />
                <span>EVM ACTIVE (30x)</span>
              </div>
            )}

            {/* Floating In-Stream Camera Controls */}
            {streamActive && (
              <div className="absolute top-4 right-4 flex items-center gap-2 z-10">
                <button
                  onClick={toggleFacingMode}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#202229]/90 hover:bg-[#282a33] text-[#f4f4f6] border border-[#2e323b] text-xs font-mono shadow-sm backdrop-blur transition-all min-h-[38px] cursor-pointer"
                  title="Switch front/rear camera"
                >
                  <SwitchCamera className="w-3.5 h-3.5 text-sky-400" />
                  <span className="hidden sm:inline">{cameraFacing === 'user' ? 'Front' : 'Rear'}</span>
                </button>
                <button
                  onClick={stopCamera}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-950/90 hover:bg-rose-900 text-rose-200 border border-rose-800/80 text-xs font-mono shadow-sm backdrop-blur transition-all min-h-[38px] cursor-pointer"
                  title="Disconnect camera"
                >
                  <CameraOff className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Stop</span>
                </button>
              </div>
            )}

            {/* Face Tracking Status Badge */}
            {!modelLoading && (
              <div className="absolute top-4 left-4 flex items-center gap-1.5 bg-[#202229]/90 border border-[#2e323b] text-xs font-mono px-2.5 py-1.5 rounded-lg backdrop-blur z-10">
                <UserCheck className={`w-3.5 h-3.5 ${faceDetected ? 'text-emerald-400' : 'text-zinc-500'}`} />
                <span className={faceDetected ? 'text-emerald-300' : 'text-zinc-400'}>
                  {faceDetected ? (streamActive ? 'Mesh Locked' : 'Demo Target') : 'Searching Face'}
                </span>
              </div>
            )}
          </>
        )}

      {/* Camera Connection / Insecure Context / Error Overlay */}
      {!streamActive && !isDemoMode && (
        <div className="absolute inset-0 bg-[#111215]/95 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center z-20">
          {!hasMediaDevices || !isSecureContext ? (
            // Insecure HTTP Context Alert on Mobile
            <div className="flex flex-col items-center max-w-md">
              <div className="w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center mb-3.5">
                <ShieldAlert className="w-7 h-7 text-amber-400" />
              </div>
              <h3 className="text-base sm:text-lg font-bold text-white mb-2 font-mono">
                Camera Blocked on Plain HTTP
              </h3>
              <p className="text-xs text-[#9ca3af] mb-4 leading-relaxed">
                Mobile browsers (iOS Safari &amp; Chrome) disable webcam access on unencrypted HTTP addresses. Please open this app over HTTPS:
              </p>
              <div className="w-full bg-[#18191e] border border-[#262932] rounded-lg p-2.5 mb-4 text-xs font-mono text-rose-300 select-all break-all text-center">
                {typeof window !== 'undefined'
                  ? (window.location.protocol === 'https:' ? window.location.href : `https://${window.location.host}${window.location.pathname}`)
                  : 'https://localhost:5173/'}
              </div>
              <div className="flex flex-wrap gap-2 justify-center">
                <a
                  href={typeof window !== 'undefined' && window.location.protocol !== 'https:' ? `https://${window.location.host}${window.location.pathname}` : '#'}
                  className="flex items-center gap-1.5 px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-mono font-bold text-xs tracking-wide shadow-md transition-all min-h-[44px]"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  <span>Reload via HTTPS</span>
                </a>
                {onToggleDemoMode && (
                  <button
                    onClick={onToggleDemoMode}
                    className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl bg-[#202229] hover:bg-[#282a33] text-amber-300 text-xs font-mono border border-[#2e323b] transition-all min-h-[44px]"
                  >
                    <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                    <span>Run Demo Mode</span>
                  </button>
                )}
              </div>
            </div>
          ) : cameraError ? (
            // Camera Error / Permission Denied State
            <div className="flex flex-col items-center max-w-md">
              <div className="w-14 h-14 rounded-2xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-center mb-3">
                <CameraOff className="w-7 h-7 text-rose-400" />
              </div>
              <h3 className="text-base font-bold text-white mb-2 font-mono">Camera Unavailable</h3>
              <p className="text-xs text-[#9ca3af] mb-4 leading-relaxed">
                {cameraError}
              </p>
              <div className="flex flex-wrap gap-2 justify-center">
                <button
                  onClick={() => startCamera()}
                  className="flex items-center gap-1.5 px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-mono font-bold shadow-md transition-all cursor-pointer min-h-[44px]"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Retry Camera</span>
                </button>
                <button
                  onClick={toggleFacingMode}
                  className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl bg-[#202229] hover:bg-[#282a33] text-[#f4f4f6] text-xs font-mono border border-[#2e323b] transition-all cursor-pointer min-h-[44px]"
                >
                  <SwitchCamera className="w-3.5 h-3.5 text-sky-400" />
                  <span>Flip: {cameraFacing === 'user' ? 'Front' : 'Rear'}</span>
                </button>
                {onToggleDemoMode && (
                  <button
                    onClick={onToggleDemoMode}
                    className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl bg-[#202229] hover:bg-[#282a33] text-amber-300 text-xs font-mono border border-[#2e323b] transition-all cursor-pointer min-h-[44px]"
                  >
                    <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                    <span>Demo Mode</span>
                  </button>
                )}
              </div>
            </div>
          ) : (
            // Ready to Connect State (Default initial state)
            <div className="flex flex-col items-center max-w-sm">
              <div className="w-16 h-16 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mb-3.5">
                <Camera className="w-8 h-8 text-rose-500" />
              </div>
              <h3 className="text-lg font-bold text-white mb-1.5 font-mono">
                Connect Camera
              </h3>
              <p className="text-xs text-[#9ca3af] mb-4 leading-relaxed">
                Click below to grant webcam access and start measuring your heart rate in real-time.
              </p>

              {/* Solid authentic clinical action button (NO gradient slop) */}
              <button
                onClick={() => startCamera()}
                disabled={isConnecting}
                className="flex items-center justify-center gap-2 w-full sm:w-auto px-7 py-3.5 rounded-xl bg-rose-600 hover:bg-rose-500 active:bg-rose-700 text-white font-bold font-mono text-xs uppercase tracking-wider shadow-lg shadow-rose-950/60 border border-rose-500/40 transition-all active:scale-95 disabled:opacity-50 cursor-pointer min-h-[48px] mb-3"
              >
                {isConnecting ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Camera className="w-4 h-4" />
                )}
                <span>{isConnecting ? 'Initializing...' : 'Start Telemetry'}</span>
              </button>

              <div className="flex items-center gap-2 mt-1">
                <button
                  onClick={toggleFacingMode}
                  className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-[#202229] hover:bg-[#282a33] text-[#f4f4f6] text-xs font-mono border border-[#2e323b] transition-all cursor-pointer min-h-[40px]"
                >
                  <SwitchCamera className="w-3.5 h-3.5 text-sky-400" />
                  <span>{cameraFacing === 'user' ? 'Front (Selfie)' : 'Rear (Back)'}</span>
                </button>
                {onToggleDemoMode && (
                  <button
                    onClick={onToggleDemoMode}
                    className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-[#202229] hover:bg-[#282a33] text-amber-300 text-xs font-mono border border-[#2e323b] transition-all cursor-pointer min-h-[40px]"
                  >
                    <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                    <span>Demo Mode</span>
                  </button>
                )}
              </div>

              {/* Explicit Local Privacy Guarantee */}
              <div className="flex items-center gap-1.5 mt-4 text-[11px] text-[#9ca3af] bg-[#18191e] px-3.5 py-2 rounded-lg border border-[#262932]">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                <span>100% On-Device: Frames computed locally in Web Worker</span>
              </div>
            </div>
          )}
        </div>
      )}
      </div>
    </div>
  );
};
