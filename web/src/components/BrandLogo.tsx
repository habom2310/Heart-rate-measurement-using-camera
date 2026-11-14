import React from 'react';

interface BrandLogoProps {
  className?: string;
  size?: number;
}

export const BrandLogo: React.FC<BrandLogoProps> = ({ className = '', size = 32 }) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 32 32"
      width={size}
      height={size}
      fill="none"
      className={`shrink-0 ${className}`}
      aria-hidden="true"
    >
      {/* Precision Instrument Housing */}
      <rect
        width="32"
        height="32"
        rx="7"
        fill="#141519"
        stroke="#2e323b"
        strokeWidth="1.2"
      />

      {/* Optical Camera Viewfinder Reticles */}
      <path
        d="M 6 11 V 7.5 A 1.5 1.5 0 0 1 7.5 6 H 11"
        stroke="#4b5563"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M 26 11 V 7.5 A 1.5 1.5 0 0 0 24.5 6 H 21"
        stroke="#4b5563"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M 6 21 V 24.5 A 1.5 1.5 0 0 0 7.5 26 H 11"
        stroke="#4b5563"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M 26 21 V 24.5 A 1.5 1.5 0 0 1 24.5 26 H 21"
        stroke="#4b5563"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Arterial Hemodynamic Pulse Wave (BVP) with Dicrotic Notch */}
      <path
        d="M 4 16 H 9.5 L 11.5 18.5 L 15 7.5 L 18 22.5 L 20.5 13.5 L 22.5 16 H 28"
        stroke="#e11d48"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Systolic Ejection Apex Highlight */}
      <circle cx="15" cy="7.5" r="1.25" fill="#fb7185" />
    </svg>
  );
};
