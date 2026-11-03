export type UploadProgress = {
  uploadedBytes: number;
  totalBytes: number;
  completedFiles: number;
  totalFiles: number;
};

export default function UploadProgressToast(props: UploadProgress) {
  const max = props.totalBytes || 1;
  const value = props.totalBytes ? props.uploadedBytes : 1;

  return (
    <div className="toast toast-end z-50">
      <output className="alert alert-info flex w-80 max-w-[calc(100vw-2rem)] flex-col items-stretch gap-2">
        <span>
          Uploading files ({props.completedFiles}/{props.totalFiles})
        </span>
        <progress
          className="progress progress-primary w-full"
          value={value}
          max={max}
          aria-label="File upload progress"
          aria-valuetext={`${Math.round((100 * value) / max)}%`}
        />
      </output>
    </div>
  );
}
