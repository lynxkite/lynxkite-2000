function formatTime(secondsStr: number | string | undefined | null): string {
  const seconds = Number(secondsStr);
  if (!seconds || !Number.isFinite(seconds) || seconds < 0) return "00:00";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

export function NodeProgress({
  telemetry,
  color,
  status,
}: {
  telemetry: any;
  color: string;
  status?: string;
}) {
  if (!telemetry) return null;

  const isMap = telemetry instanceof Map || typeof telemetry.entries === "function";
  const t = isMap
    ? Object.fromEntries(telemetry.entries ? telemetry.entries() : telemetry)
    : telemetry;

  if (Object.keys(t).length === 0) return null;

  const { n = 0, total = 0, elapsed = 0, rate, prefix, postfix, unit = "it", colour } = t;

  const hasTotal = typeof total === "number" && total > 0;
  // If we don't have a total, we can show an indeterminate animated bar or full bar
  const isIndeterminate = !hasTotal && status === "active";
  const percentage = hasTotal ? Math.min(100, Math.max(0, (n / total) * 100)) : 100;

  // Estimate time remaining
  const eta = hasTotal && rate && rate > 0 ? (total - n) / rate : null;
  const itersPerSec =
    typeof rate === "number" && Number.isFinite(rate)
      ? rate < 1
        ? `${(1 / rate).toFixed(2)} s/${unit}`
        : `${rate.toFixed(2)} ${unit}/s`
      : "";

  return (
    <div
      className="node-progress"
      style={{
        padding: "8px 12px",
        fontSize: "12px",
        background: "var(--bg-secondary, rgba(0,0,0,0.05))",
        borderBottom: "1px solid var(--border-color, rgba(128,128,128,0.2))",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
        <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>
          {prefix || "Execution Progress"}
        </span>
        <span style={{ fontWeight: 500 }}>
          {hasTotal ? `${percentage.toFixed(1)}%` : `${n} ${unit}`}
        </span>
      </div>

      <div
        className={isIndeterminate ? "progress-indeterminate" : ""}
        style={{
          height: "6px",
          background: "rgba(128, 128, 128, 0.2)",
          borderRadius: "3px",
          overflow: "hidden",
          marginBottom: "6px",
          position: "relative",
        }}
      >
        <div
          className={isIndeterminate ? "indeterminate-bar" : ""}
          style={{
            height: "100%",
            width: isIndeterminate ? "100%" : `${percentage}%`,
            backgroundColor: colour || color,
            transition: "width 0.1s linear",
            opacity: 0.9,
          }}
        />
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          color: "var(--text-secondary, #888)",
          gap: "8px",
        }}
      >
        <span
          title={postfix ? `${n} ${hasTotal ? `/ ${total} ` : ""}${unit} — ${postfix}` : undefined}
          style={{
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            flex: 1,
          }}
        >
          {n} {hasTotal ? `/ ${total}` : ""} {unit} {postfix ? `— ${postfix}` : ""}
        </span>
        <span style={{ display: "flex", gap: "8px", textAlign: "right" }}>
          {itersPerSec && <span>{itersPerSec}</span>}
          <span>
            {eta !== null && eta > 0
              ? `ETA: ${formatTime(eta)}`
              : `Elapsed: ${formatTime(elapsed)}`}
          </span>
        </span>
      </div>
    </div>
  );
}
