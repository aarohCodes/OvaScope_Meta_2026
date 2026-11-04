"use client";

import { useRef, useState } from "react";

interface Props {
  maxBytes: number;
  disabled?: boolean;
  onFile: (file: File) => void;
}

const ACCEPT = ["image/png", "image/jpeg"];

export function UploadDropzone({ maxBytes, disabled, onFile }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const take = (f: File | undefined) => {
    if (!f) return;
    // Convenience checks only; the server re-validates decoded content.
    if (!ACCEPT.includes(f.type)) {
      setProblem("Only PNG and JPEG still images are supported. DICOM and video are not supported in this version.");
      return;
    }
    if (f.size > maxBytes) {
      setProblem(`File exceeds the ${Math.round(maxBytes / 1024 / 1024)} MB limit.`);
      return;
    }
    setProblem(null);
    onFile(f);
  };

  return (
    <div>
      <div
        data-testid="dropzone"
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (!disabled) take(e.dataTransfer.files?.[0]);
        }}
        className={`flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-14 text-center transition-colors ${
          over ? "border-accent-mark bg-accent-deep/10" : "border-line-strong bg-surface"
        }`}
      >
        <p className="text-base text-ink">Drop a PNG or JPEG ultrasound still here</p>
        <p className="text-sm text-muted">or</p>
        <button
          type="button"
          disabled={disabled}
          onClick={() => input.current?.click()}
          className="rounded-md bg-accent-deep px-4 py-2 text-sm font-medium text-white hover:bg-accent-mark disabled:opacity-50"
        >
          Choose file
        </button>
        <input
          ref={input}
          data-testid="file-input"
          type="file"
          accept="image/png,image/jpeg"
          className="sr-only"
          aria-label="Choose an ultrasound image file"
          onChange={(e) => take(e.target.files?.[0])}
        />
        <p className="text-xs text-muted">Max {Math.round(maxBytes / 1024 / 1024)} MB · still images only</p>
      </div>
      {problem && (
        <p role="alert" className="mt-3 rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-danger">
          {problem}
        </p>
      )}
    </div>
  );
}
