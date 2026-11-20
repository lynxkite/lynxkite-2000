export type DownloadAction = "download" | "export";

export type DownloadProgress = {
  phase: "preparing" | "downloading";
  receivedBytes: number;
  totalBytes: number | null;
};

export type PendingDownload = DownloadProgress & {
  action: DownloadAction;
  itemName: string;
};

function formatBytes(bytes: number) {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

export default function DownloadProgressToast(props: { pendingDownload: PendingDownload }) {
  const { pendingDownload } = props;

  return (
    <div className="toast toast-end z-50">
      <output
        className="alert alert-info flex w-80 max-w-[calc(100vw-2rem)] flex-col items-stretch gap-2"
        aria-live="polite"
      >
        <span>
          {pendingDownload.phase === "preparing"
            ? `Preparing ${pendingDownload.itemName}...`
            : `Downloading ${pendingDownload.itemName}`}
        </span>
        {pendingDownload.phase === "preparing" ? (
          <progress className="progress progress-primary w-full" aria-label="Preparing download" />
        ) : (
          <>
            <progress
              className="progress progress-primary w-full"
              value={
                pendingDownload.totalBytes === null ? undefined : pendingDownload.receivedBytes
              }
              max={pendingDownload.totalBytes ?? 1}
              aria-label="Download progress"
              aria-valuetext={
                pendingDownload.totalBytes === null
                  ? `${formatBytes(pendingDownload.receivedBytes)} downloaded`
                  : `${Math.min(
                      100,
                      Math.round(
                        (100 * pendingDownload.receivedBytes) / pendingDownload.totalBytes,
                      ),
                    )}%`
              }
            />
            <span className="flex justify-between gap-2 text-sm tabular-nums">
              {formatBytes(pendingDownload.receivedBytes)}
              {pendingDownload.totalBytes !== null && (
                <>
                  <span>of {formatBytes(pendingDownload.totalBytes)}</span>
                  <span>
                    {Math.min(
                      100,
                      Math.round(
                        (100 * pendingDownload.receivedBytes) / pendingDownload.totalBytes,
                      ),
                    )}
                    %
                  </span>
                </>
              )}
            </span>
          </>
        )}
      </output>
    </div>
  );
}
